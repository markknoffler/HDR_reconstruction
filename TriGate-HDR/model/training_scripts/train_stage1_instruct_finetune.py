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
import csv
import os
import time

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..instructpix2pix_pretrained_finetuned_stage1 import TrainableTriGateInstructPix2PixStage1
from .common_training import (
    HDRVDPMetrics,
    add_subset_args,
    apply_smoke_test_args,
    default_hrishav_data_paths,
    maybe_resume,
    sanitize_data_path,
    print_epoch_summary,
    save_best_checkpoint,
    save_checkpoint,
    save_latest_checkpoint,
)
from .dataset_splits import build_dataloaders
from .val_export import (
    export_final_test_samples,
    make_stage1_instruct_predictor,
    validate_model_mtraining,
)


def _append_extended_metrics_csv(
    csv_path: str,
    epoch: int,
    train_loss: float,
    tr_psnr: float,
    tr_ssim: float,
    tr_hdrvdp2: float,
    tr_hdrvdp3: float,
    val_psnr,
    val_ssim,
    val_hdrvdp2,
    val_hdrvdp3,
    val_ran: bool,
):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = [
            "epoch",
            "train_loss",
            "train_subset_psnr",
            "train_subset_ssim",
            "train_subset_hdrvdp2",
            "train_subset_hdrvdp3",
            "val_psnr",
            "val_ssim",
            "val_hdrvdp2",
            "val_hdrvdp3",
            "val_ran",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{float(train_loss):.6f}",
                "train_subset_psnr": f"{float(tr_psnr):.4f}",
                "train_subset_ssim": f"{float(tr_ssim):.4f}",
                "train_subset_hdrvdp2": f"{float(tr_hdrvdp2):.4f}",
                "train_subset_hdrvdp3": f"{float(tr_hdrvdp3):.4f}",
                "val_psnr": f"{float(val_psnr):.4f}" if val_ran else "",
                "val_ssim": f"{float(val_ssim):.4f}" if val_ran else "",
                "val_hdrvdp2": f"{float(val_hdrvdp2):.4f}" if val_ran else "",
                "val_hdrvdp3": f"{float(val_hdrvdp3):.4f}" if val_ran else "",
                "val_ran": int(val_ran),
            }
        )


def _make_train_subset_loader(train_loader, sample_count: int, num_workers: int, seed: int):
    ds = train_loader.dataset
    indices = list(getattr(ds, "indices", []))
    if not indices:
        return None
    if len(indices) <= sample_count:
        picked = indices
    else:
        g = torch.Generator()
        g.manual_seed(int(seed))
        order = torch.randperm(len(indices), generator=g).tolist()
        picked = [indices[i] for i in order[:sample_count]]
    subset = Subset(ds.dataset, picked)
    return DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def main():
    _defaults = default_hrishav_data_paths()
    parser = argparse.ArgumentParser(
        description="Fine-tune Stage-1 InstructPix2Pix + TriGate encoders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_lr", type=float, default=1e-4)
    parser.add_argument("--encoder_lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default=_defaults["checkpoint_dir"])
    parser.add_argument("--ldr_dir", type=str, default=_defaults["ldr_dir"])
    parser.add_argument("--hdr_dir", type=str, default=_defaults["hdr_dir"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sam_mask_dir", type=str, default=_defaults["sam_mask_dir"])
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--continue_train",
        action="store_true",
        help="Resume from checkpoint_dir/latest.pt (epoch, weights, optimizer).",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default="",
        help="Optional explicit .pt (e.g. experiments/stage1_instruct/epoch_5.pt). "
        "Overrides latest.pt when used with --continue_train.",
    )
    parser.add_argument("--model_id", type=str, default="timbrooks/instruct-pix2pix")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--diffusion_only_epochs", type=int, default=5)
    parser.add_argument("--novelty_ramp_epochs", type=int, default=10)
    parser.add_argument("--max_novelty_weight", type=float, default=0.25)
    parser.add_argument("--val_inference_steps", type=int, default=20)
    parser.add_argument(
        "--full_val_every",
        type=int,
        default=10,
        help="Run full validation split every N epochs (and at last epoch).",
    )
    parser.add_argument(
        "--train_eval_samples",
        type=int,
        default=100,
        help="Number of train images used each epoch for train-set metric probe.",
    )
    parser.add_argument("--use_real_hdrvdp", action="store_true")
    parser.add_argument(
        "--no_vae_slicing",
        action="store_true",
        help="Disable VAE slice/tiling (uses more VRAM during novelty decode).",
    )
    parser.add_argument(
        "--clear_cuda_cache_each_step",
        action="store_true",
        help="torch.cuda.empty_cache() each train step when novelty loss is active.",
    )
    add_subset_args(parser)
    args = parser.parse_args()
    args = apply_smoke_test_args(args)

    args.ldr_dir = sanitize_data_path(args.ldr_dir)
    args.hdr_dir = sanitize_data_path(args.hdr_dir)
    args.sam_mask_dir = sanitize_data_path(args.sam_mask_dir) if args.sam_mask_dir else ""
    args.checkpoint_dir = sanitize_data_path(args.checkpoint_dir)
    args.resume_from = sanitize_data_path(args.resume_from) if args.resume_from else ""

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
    model = model.to(device)
    if args.no_vae_slicing and hasattr(model.vae, "disable_slicing"):
        try:
            model.vae.disable_slicing()
            model.vae.disable_tiling()
        except Exception:
            pass

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
    print(f"  Checkpoints     : {args.checkpoint_dir}")
    if args.continue_train:
        start_epoch, best_psnr, best_ssim, best_hdrvdp2, best_hdrvdp3 = maybe_resume(
            args.checkpoint_dir,
            model,
            optimizer,
            resume_from=args.resume_from,
        )
        model = model.to(device)
        print(
            f"Resuming training at epoch {start_epoch}/{args.epochs} "
            f"(best PSNR={best_psnr:.4f}, SSIM={best_ssim:.4f})"
        )

    print(f"  LDR dir         : {args.ldr_dir}")
    print(f"  HDR dir         : {args.hdr_dir}")
    print(f"  SAM masks       : {args.sam_mask_dir}")

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
        num_workers=args.num_workers,
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
    print(f"  Full validation : every {args.full_val_every} epochs (+ final epoch)")
    print(f"  Train probe     : {args.train_eval_samples} train images per epoch")
    if args.torch_dtype == "float32" and not args.amp:
        print(
            "  [VRAM] float32 without --amp uses ~2x memory. "
            f"Novelty VAE decode starts after epoch {args.diffusion_only_epochs} "
            "(often OOM on 20GB). Try: --amp --train_eval_samples 20"
        )

    predict_fn = make_stage1_instruct_predictor(model, num_inference_steps=args.val_inference_steps)

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch <= args.diffusion_only_epochs:
            model.set_training_phase(1)
        else:
            model.set_training_phase(2)

        novelty_w_ep = 0.0
        if epoch > args.diffusion_only_epochs:
            from ..instructpix2pix_pretrained_finetuned_stage1.losses import curriculum_novelty_weight

            novelty_w_ep = curriculum_novelty_weight(
                epoch,
                args.diffusion_only_epochs,
                args.novelty_ramp_epochs,
                args.max_novelty_weight,
            )
        if novelty_w_ep > 0 and epoch == args.diffusion_only_epochs + 1:
            print(
                f"\n[VRAM] Epoch {epoch}+: novelty loss active — extra full VAE decode each step. "
                "VAE slice/tiling + UNet grad-checkpoint enabled.\n"
            )

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
            if (
                args.clear_cuda_cache_each_step
                and device.type == "cuda"
                and float(parts.get("novelty_weight", torch.tensor(0.0))) > 0
            ):
                torch.cuda.empty_cache()
            running += loss.item()
            diff_v = parts.get("diffusion", torch.tensor(0.0))
            pbar.set_postfix(
                loss=f"{running / step:.4f}",
                diff=f"{float(diff_v):.4f}",
                nov_w=f"{float(parts.get('novelty_weight', 0)):.3f}",
            )

        train_loss = running / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start

        model.eval()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        train_probe_loader = _make_train_subset_loader(
            train_loader,
            sample_count=max(1, int(args.train_eval_samples)),
            num_workers=args.num_workers,
            seed=args.val_export_seed + epoch,
        )
        tr_psnr, tr_ssim, tr_hdrvdp2, tr_hdrvdp3 = 0.0, 0.0, 0.0, 0.0
        if train_probe_loader is not None:
            n_probe = len(train_probe_loader.dataset)
            n_train = len(train_loader.dataset)
            print(
                f"Train-probe metrics on {n_probe} samples "
                f"(requested {args.train_eval_samples}, train split size {n_train})..."
            )
            tr_psnr, tr_ssim, tr_hdrvdp2, tr_hdrvdp3 = validate_model_mtraining(
                train_probe_loader,
                device,
                epoch,
                hdrvdp_calculator,
                predict_fn,
                validation_root,
                save_samples=False,
                max_samples=0,
                amp=args.amp,
            )
            print(
                f"  Train-probe PSNR/SSIM/HDRVDP2/HDRVDP3: "
                f"{tr_psnr:.4f} / {tr_ssim:.4f} / {tr_hdrvdp2:.4f} / {tr_hdrvdp3:.4f}"
            )

        val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = 0.0, 0.0, 0.0, 0.0
        val_ran = False
        do_full_val = (
            val_loader is not None
            and (
                epoch % max(1, int(args.full_val_every)) == 0
                or epoch == args.epochs
            )
        )
        if do_full_val:
            print("Running full validation split...")
            save_val_images = args.save_val_samples_each_epoch
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
            val_ran = True
            print(
                f"  Full-val PSNR/SSIM/HDRVDP2/HDRVDP3: "
                f"{val_psnr:.4f} / {val_ssim:.4f} / {val_hdrvdp2:.4f} / {val_hdrvdp3:.4f}"
            )
        model.train()

        _append_extended_metrics_csv(
            csv_path,
            epoch,
            train_loss,
            tr_psnr,
            tr_ssim,
            tr_hdrvdp2,
            tr_hdrvdp3,
            val_psnr,
            val_ssim,
            val_hdrvdp2,
            val_hdrvdp3,
            val_ran=val_ran,
        )
        print(f"  Metrics appended to: {csv_path}")

        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_subset_psnr": tr_psnr,
            "train_subset_ssim": tr_ssim,
            "train_subset_hdrvdp2": tr_hdrvdp2,
            "train_subset_hdrvdp3": tr_hdrvdp3,
            "val_psnr": val_psnr if val_ran else None,
            "val_ssim": val_ssim if val_ran else None,
            "val_hdrvdp2": val_hdrvdp2 if val_ran else None,
            "val_hdrvdp3": val_hdrvdp3 if val_ran else None,
            "val_ran": val_ran,
            "best_val_psnr": best_psnr,
            "best_val_ssim": best_ssim,
            "best_val_hdrvdp2": best_hdrvdp2,
            "best_val_hdrvdp3": best_hdrvdp3,
            "stage1_type": "instruct_pix2pix_finetune",
            "model_id": args.model_id,
        }

        save_latest_checkpoint(args.checkpoint_dir, payload)
        print(f"  Saved latest.pt (resume with --continue_train)")

        if val_ran and val_psnr > best_psnr:
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

        if val_ran and args.save_val_samples_each_epoch:
            print(f"  Validation samples: {os.path.join(validation_root, f'epoch_{epoch}')}")

        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs} Summary")
        print(f"{'=' * 60}")
        print(f"  Training Loss                : {train_loss:.6f}")
        print(
            f"  Train Probe [PSNR/SSIM/H2/H3]: "
            f"{tr_psnr:.4f} / {tr_ssim:.4f} / {tr_hdrvdp2:.4f} / {tr_hdrvdp3:.4f}"
        )
        if val_ran:
            print(
                f"  Full Val   [PSNR/SSIM/H2/H3]: "
                f"{val_psnr:.4f} / {val_ssim:.4f} / {val_hdrvdp2:.4f} / {val_hdrvdp3:.4f}"
            )
        else:
            print(f"  Full Val                    : skipped (every {args.full_val_every} epochs)")
        print(f"  Epoch Time                  : {epoch_time:.2f} seconds")
        print(f"{'=' * 60}")

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
