import argparse
import os
import time

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import Stage1TriEncoderDiffusionSystem
from ..losses.stage_composite_losses import stage1_loss
from ..losses.codebook_losses import kl_codebook_loss
from .common_training import (
    HDRVDPMetrics,
    add_subset_args,
    apply_smoke_test_args,
    maybe_resume,
    print_epoch_summary,
    save_best_checkpoint,
    save_checkpoint,
    save_metrics_to_csv,
)
from .dataset_splits import build_dataloaders
from .val_export import make_stage1_predictor, validate_model_mtraining


def main():
    parser = argparse.ArgumentParser(description="Train Stage-1 tri-encoder diffusion.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage1")
    parser.add_argument("--ldr_dir", type=str, required=False, default="")
    parser.add_argument("--hdr_dir", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sam_mask_dir", type=str, default="")
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--continue_train", action="store_true")
    add_subset_args(parser)
    args = parser.parse_args()
    args = apply_smoke_test_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    validation_root = args.validation_results_dir or os.path.join(args.checkpoint_dir, "validation_results")
    os.makedirs(validation_root, exist_ok=True)

    model = Stage1TriEncoderDiffusionSystem().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    hdrvdp_calculator = HDRVDPMetrics(use_real_hdrvdp=False)

    start_epoch = 1
    best_psnr, best_ssim = 0.0, 0.0
    best_hdrvdp2, best_hdrvdp3 = 0.0, 0.0
    if args.continue_train:
        start_epoch, best_psnr, best_ssim, best_hdrvdp2, best_hdrvdp3 = maybe_resume(
            args.checkpoint_dir, model, optimizer
        )

    train_loader, val_loader = None, None
    if args.ldr_dir and args.hdr_dir:
        train_loader, val_loader, _, _ = build_dataloaders(
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
            f"[Stage1] train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0} "
            f"smoke_test={args.smoke_test} packet={args.subset_packet} fraction={args.subset_fraction}"
        )

    csv_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")
    predict_fn = make_stage1_predictor(model)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}/{args.epochs}", leave=True)
        for step, batch in enumerate(pbar, start=1):
            ldr = batch["ldr_image"].to(device)
            hdr = batch["hdr_image"].to(device)
            segmap = batch.get("segmap", ldr).to(device)
            sam_class_masks = batch.get("sam_class_masks")
            if sam_class_masks is not None:
                sam_class_masks = sam_class_masks.to(device)
            t = torch.randint(0, 100, (ldr.shape[0],), device=device).long()
            with autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred, gate, class_probs, aux = model(ldr, t, segmap=segmap)
                loss, _ = stage1_loss(pred, hdr, gate, class_masks=sam_class_masks, class_probs=class_probs)
                loss = loss + 0.01 * kl_codebook_loss(aux["mus"], aux["logvars"])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()
            pbar.set_postfix(loss=f"{running / step:.4f}")

        train_loss = running / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start

        val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = 0.0, 0.0, 0.0, 0.0
        if val_loader is not None:
            print("Validating...")
            model.eval()
            val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = validate_model_mtraining(
                val_loader,
                device,
                epoch,
                hdrvdp_calculator,
                predict_fn,
                validation_root,
                save_samples=args.save_val_samples_each_epoch,
                max_samples=args.val_export_count,
                amp=args.amp,
            )
            model.train()

        save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3)
        print_epoch_summary(
            epoch, args.epochs, train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3, epoch_time
        )

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
        }

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

        if epoch % args.save_ckpt_after == 0:
            save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", payload)
            print(f"  Saved checkpoint at epoch {epoch}")

        if args.save_val_samples_each_epoch:
            print(f"  Validation samples: {os.path.join(validation_root, f'epoch_{epoch}')}")


if __name__ == "__main__":
    main()
