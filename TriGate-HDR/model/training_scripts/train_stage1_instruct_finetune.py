"""
Train Stage-1: InstructPix2Pix (pretrained) + UNet LoRA + TriGate encoder conditioning.

Training loop aligned with ARThdrNet/m_training.py and train_stage1_dual_diffusion.py:
  - Every epoch: full-val PSNR, SSIM, HDR-VDP-2/3 -> training_metrics.csv
  - Every epoch: validation_results/epoch_{N}/ (LDR, pred HDR, gt HDR)
  - Every epoch: latest.pt (for --continue_train)
  - Every save_ckpt_after epochs (default 5): epoch_{N}.pt
  - On best PSNR: best.pt + best_epoch_{N}.pt
  - After all epochs: final_test_exports/ with N random val LDR->HDR samples
"""

import argparse
import os
import time

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..instructpix2pix_pretrained_finetuned_stage1 import TrainableTriGateInstructPix2PixStage1
from .common_training import (
    HDRVDPMetrics,
    add_subset_args,
    apply_smoke_test_args,
    maybe_resume,
    print_epoch_summary,
    save_best_checkpoint,
    save_checkpoint,
    save_latest_checkpoint,
    save_metrics_to_csv,
)
from .dataset_splits import build_dataloaders
from .val_export import (
    export_final_test_samples,
    make_stage1_instruct_predictor,
    validate_model_mtraining,
)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stage-1 InstructPix2Pix + TriGate encoders.")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage1_instruct")
    parser.add_argument("--ldr_dir", type=str, required=False, default="")
    parser.add_argument("--hdr_dir", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sam_mask_dir", type=str, default="")
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--continue_train",
        action="store_true",
        help="Resume from checkpoint_dir/latest.pt (epoch, weights, optimizer).",
    )
    parser.add_argument("--model_id", type=str, default="timbrooks/instruct-pix2pix")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--diffusion_only_epochs", type=int, default=5)
    parser.add_argument("--novelty_ramp_epochs", type=int, default=10)
    parser.add_argument("--max_novelty_weight", type=float, default=0.25)
    parser.add_argument("--val_inference_steps", type=int, default=20)
    parser.add_argument("--use_real_hdrvdp", action="store_true")
    add_subset_args(parser)
    args = parser.parse_args()
    args = apply_smoke_test_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    validation_root = args.validation_results_dir or os.path.join(args.checkpoint_dir, "validation_results")
    os.makedirs(validation_root, exist_ok=True)
    val_export_dir = args.val_export_dir or os.path.join(args.checkpoint_dir, "val_exports")
    final_test_dir = os.path.join(val_export_dir, "final_test_exports")
    csv_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")

    print(f"[Stage1-Instruct] Loading {args.model_id} (dtype={args.torch_dtype})...")
    model = TrainableTriGateInstructPix2PixStage1.from_pretrained(
        model_id=args.model_id,
        device=device,
        torch_dtype=args.torch_dtype,
        lora_rank=args.lora_rank,
    )

    lora_params = [p for n, p in model.unet.named_parameters() if p.requires_grad]
    enc_params = [p for p in model.cond_injector.parameters() if p.requires_grad]
    param_groups = []
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lora_lr})
    if enc_params:
        param_groups.append({"params": enc_params, "lr": args.encoder_lr})
    optimizer = optim.AdamW(param_groups if param_groups else model.parameters(), lr=args.lr)

    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    hdrvdp_calculator = HDRVDPMetrics(use_real_hdrvdp=args.use_real_hdrvdp)

    start_epoch = 1
    best_psnr, best_ssim = 0.0, 0.0
    best_hdrvdp2, best_hdrvdp3 = 0.0, 0.0
    if args.continue_train:
        start_epoch, best_psnr, best_ssim, best_hdrvdp2, best_hdrvdp3 = maybe_resume(
            args.checkpoint_dir, model, optimizer
        )
        print(
            f"Resuming from epoch {start_epoch} "
            f"(best PSNR={best_psnr:.4f}, SSIM={best_ssim:.4f})"
        )

    if not args.ldr_dir or not args.hdr_dir:
        raise SystemExit("Provide --ldr_dir and --hdr_dir (or use --smoke_test with defaults).")

    train_loader, val_loader, full_dataset, val_indices = build_dataloaders(
        args.ldr_dir,
        args.hdr_dir,
        args.batch_size,
        sam_mask_dir=args.sam_mask_dir,
        max_sam_classes=args.max_sam_classes,
        max_dim=args.max_dim,
        val_ratio=args.val_ratio,
        split_seed=args.split_seed,
        subset_fraction=args.subset_fraction,
        subset_packet=args.subset_packet,
        checkpoint_dir=args.checkpoint_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    print(
        f"[Stage1-Instruct] train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset) if val_loader else 0} "
        f"smoke_test={args.smoke_test} save_ckpt_every={args.save_ckpt_after}"
    )
    print(f"  Metrics CSV     : {csv_path}")
    print(f"  Validation dumps: {validation_root}/epoch_<N>/")
    print(f"  Checkpoints     : latest.pt every epoch; epoch_<N>.pt every {args.save_ckpt_after} epochs")

    predict_fn = make_stage1_instruct_predictor(model, num_inference_steps=args.val_inference_steps)

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.diffusion_only_epochs:
            model.set_training_phase(1)
        else:
            model.set_training_phase(2)

        epoch_start = time.time()
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Instruct Stage1 {epoch}/{args.epochs}", leave=True)
        for step, batch in enumerate(pbar, start=1):
            ldr = batch["ldr_image"].to(device)
            hdr = batch["hdr_image"].to(device)
            segmap = batch.get("segmap", ldr)
            if torch.is_tensor(segmap):
                segmap = segmap.to(device)
            sam_class_masks = batch.get("sam_class_masks")
            if sam_class_masks is not None:
                sam_class_masks = sam_class_masks.to(device)

            with autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss, parts = model.compute_training_loss(
                    ldr,
                    hdr,
                    segmap=segmap,
                    sam_class_masks=sam_class_masks,
                    epoch=epoch,
                    diffusion_only_epochs=args.diffusion_only_epochs,
                    novelty_ramp_epochs=args.novelty_ramp_epochs,
                    max_novelty_weight=args.max_novelty_weight,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            diff_v = parts.get("diffusion", torch.tensor(0.0))
            pbar.set_postfix(
                loss=f"{running / step:.4f}",
                diff=f"{float(diff_v):.4f}",
                nov_w=f"{float(parts.get('novelty_weight', 0)):.3f}",
            )

        train_loss = running / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start

        val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = 0.0, 0.0, 0.0, 0.0
        if val_loader is not None:
            print("Validating...")
            model.eval()
            # ARThdrNet: dump val images when epoch % save_ckpt_after == 0 or epoch == 1;
            # TriGate default: every epoch (user request). Cap count with val_export_count.
            save_val_images = args.save_val_samples_each_epoch or (
                epoch == 1 or epoch % args.save_ckpt_after == 0
            )
            val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = validate_model_mtraining(
                val_loader,
                device,
                epoch,
                hdrvdp_calculator,
                predict_fn,
                validation_root,
                save_samples=save_val_images,
                max_samples=args.val_export_count,
                amp=args.amp,
            )
            model.train()

        save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3)
        print_epoch_summary(
            epoch, args.epochs, train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3, epoch_time
        )
        print(f"  Metrics appended to: {csv_path}")

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_hdrvdp2": val_hdrvdp2,
            "val_hdrvdp3": val_hdrvdp3,
            "best_val_psnr": best_psnr,
            "best_val_ssim": best_ssim,
            "best_val_hdrvdp2": best_hdrvdp2,
            "best_val_hdrvdp3": best_hdrvdp3,
            "stage1_type": "instruct_pix2pix_finetune",
            "model_id": args.model_id,
        }

        save_latest_checkpoint(args.checkpoint_dir, payload)
        print(f"  Saved latest.pt (resume with --continue_train)")

        if val_psnr > best_psnr:
            best_psnr, best_ssim = val_psnr, val_ssim
            best_hdrvdp2, best_hdrvdp3 = val_hdrvdp2, val_hdrvdp3
            payload["best_val_psnr"] = best_psnr
            payload["best_val_ssim"] = best_ssim
            payload["best_val_hdrvdp2"] = best_hdrvdp2
            payload["best_val_hdrvdp3"] = best_hdrvdp3
            save_best_checkpoint(args.checkpoint_dir, payload)
            save_checkpoint(args.checkpoint_dir, f"best_epoch_{epoch}", payload)
            print(f"  Saved best model with PSNR: {val_psnr:.4f}")

        if epoch % args.save_ckpt_after == 0 or epoch == args.epochs:
            save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", payload)
            print(f"  Saved checkpoint: epoch_{epoch}.pt")

        if save_val_images:
            print(f"  Validation samples: {os.path.join(validation_root, f'epoch_{epoch}')}")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"  Best PSNR      : {best_psnr:.4f} dB")
    print(f"  Best SSIM      : {best_ssim:.4f}")
    print(f"  Best HDR-VDP-2 : {best_hdrvdp2:.4f}")
    print(f"  Best HDR-VDP-3 : {best_hdrvdp3:.4f}")
    print(f"  Metrics CSV    : {csv_path}")
    print("=" * 60)

    if not args.skip_final_test_export and val_indices:
        print("\nRunning final test export (random validation LDR -> HDR)...")
        model.eval()
        export_final_test_samples(
            full_dataset,
            val_indices,
            device,
            predict_fn,
            final_test_dir,
            count=args.final_test_count,
            seed=args.val_export_seed,
            amp=args.amp,
        )
        print(f"Final test outputs: {final_test_dir}")
    elif args.skip_final_test_export:
        print("Skipped final test export (--skip_final_test_export).")


if __name__ == "__main__":
    main()
