import argparse
import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..seaming_model.gan_system import SeamingGANSystem
from ..losses.stage_composite_losses import stage3_loss
from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import Stage1TriEncoderDiffusionSystem
from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from .common_training import (
    HDRVDPMetrics,
    add_subset_args,
    apply_smoke_test_args,
    load_checkpoint,
    print_epoch_summary,
    save_best_checkpoint,
    save_checkpoint,
    save_metrics_to_csv,
)
from .dataset_splits import build_dataloaders
from .val_export import make_stage3_predictor, validate_model_mtraining


def build_composited_input(stage2_hdr, stage1_hdr, gate):
    clip_mask = (1.0 - gate).clamp(0.0, 1.0)
    composed = stage2_hdr * (1.0 - clip_mask) + stage1_hdr * clip_mask
    dilated = F.max_pool2d(clip_mask, kernel_size=17, stride=1, padding=8)
    eroded = -F.max_pool2d(-clip_mask, kernel_size=9, stride=1, padding=4)
    seam_band = (dilated - eroded).clamp(0.0, 1.0)
    seam_band = torch.maximum(seam_band, clip_mask)
    return composed, seam_band


def _resolve_stage_ckpt(path_arg: str, default_dir: str) -> str:
    if path_arg and os.path.isfile(path_arg):
        return path_arg
    best = os.path.join(default_dir, "best.pt")
    if os.path.isfile(best):
        return best
    latest = os.path.join(default_dir, "latest.pt")
    if os.path.isfile(latest):
        return latest
    return path_arg


def main():
    parser = argparse.ArgumentParser(description="Train Stage-3 seaming GAN (full pipeline inference).")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage3")
    parser.add_argument("--ldr_dir", type=str, required=False, default="")
    parser.add_argument("--hdr_dir", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--stage1_ckpt_dir", type=str, default="checkpoints_stage1")
    parser.add_argument("--stage2_ckpt_dir", type=str, default="checkpoints_stage2")
    parser.add_argument("--sam_mask_dir", type=str, default="")
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--outside_lock_weight", type=float, default=0.5)
    parser.add_argument("--continue_train", action="store_true")
    add_subset_args(parser)
    args = parser.parse_args()
    args = apply_smoke_test_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    validation_root = args.validation_results_dir or os.path.join(args.checkpoint_dir, "validation_results")
    os.makedirs(validation_root, exist_ok=True)

    system = SeamingGANSystem().to(device)
    frozen_stage1 = Stage1TriEncoderDiffusionSystem().to(device)
    frozen_stage2 = ColdHDRDiffusion().to(device)

    stage1_path = _resolve_stage_ckpt(args.stage1_ckpt, args.stage1_ckpt_dir)
    stage2_path = _resolve_stage_ckpt(args.stage2_ckpt, args.stage2_ckpt_dir)
    if stage1_path and os.path.isfile(stage1_path):
        ckpt = load_checkpoint(stage1_path, device)
        frozen_stage1.load_state_dict(ckpt["model"], strict=False)
        print(f"[Stage3] loaded Stage1 from {stage1_path}")
    if stage2_path and os.path.isfile(stage2_path):
        ckpt = load_checkpoint(stage2_path, device)
        frozen_stage2.load_state_dict(ckpt["model"], strict=False)
        print(f"[Stage3] loaded Stage2 from {stage2_path}")

    frozen_stage1.eval()
    frozen_stage2.eval()
    for p in frozen_stage1.parameters():
        p.requires_grad = False
    for p in frozen_stage2.parameters():
        p.requires_grad = False

    opt_g = optim.Adam(system.generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(system.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    scaler_g = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    scaler_d = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    hdrvdp_calculator = HDRVDPMetrics(use_real_hdrvdp=False)

    start_epoch = 1
    best_psnr, best_ssim = 0.0, 0.0
    best_hdrvdp2, best_hdrvdp3 = 0.0, 0.0
    if args.continue_train:
        latest = os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.isfile(latest):
            ckpt = load_checkpoint(latest, device)
            system.generator.load_state_dict(ckpt["generator"], strict=False)
            system.discriminator.load_state_dict(ckpt["discriminator"], strict=False)
            opt_g.load_state_dict(ckpt["opt_g"])
            opt_d.load_state_dict(ckpt["opt_d"])
            start_epoch = ckpt["epoch"] + 1
            best_psnr = ckpt.get("best_val_psnr", 0.0)
            best_ssim = ckpt.get("best_val_ssim", 0.0)
            best_hdrvdp2 = ckpt.get("best_val_hdrvdp2", 0.0)
            best_hdrvdp3 = ckpt.get("best_val_hdrvdp3", 0.0)

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
            f"[Stage3] train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0} "
            f"smoke_test={args.smoke_test} packet={args.subset_packet} fraction={args.subset_fraction}"
        )

    csv_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")
    predict_fn = make_stage3_predictor(frozen_stage1, frozen_stage2, system.generator)

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        system.train()
        running_d, running_g = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}/{args.epochs}", leave=True)
        for step, batch in enumerate(pbar, start=1):
            ldr = batch["ldr_image"].to(device)
            gate = batch["gate"].to(device)
            gt = batch["hdr_image"].to(device)
            sam_class_masks = batch.get("sam_class_masks")
            if sam_class_masks is not None:
                sam_class_masks = sam_class_masks.to(device)

            with torch.no_grad():
                t = torch.randint(0, 100, (ldr.shape[0],), device=device).long()
                with autocast("cuda", enabled=args.amp and device.type == "cuda"):
                    gen_clip, _, class_probs, _ = frozen_stage1(ldr, t, segmap=batch.get("segmap", ldr).to(device))
                    stage2_hdr = frozen_stage2.restore_hdr(ldr, gate=gate)
                composed_x, seam_mask = build_composited_input(stage2_hdr, gen_clip, gate)

            opt_d.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp and device.type == "cuda"):
                fake, fg, fs = system(composed_x, gen_clip, seam_mask)
                rg, rs = system.discriminator(gt, seam_mask)
                d_loss = system.d_hinge_loss(rg, fg.detach()) + system.d_hinge_loss(rs, fs.detach())
            scaler_d.scale(d_loss).backward()
            scaler_d.step(opt_d)
            scaler_d.update()
            running_d += d_loss.item()

            opt_g.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp and device.type == "cuda"):
                fake, fg, fs = system(composed_x, gen_clip, seam_mask)
                recon, _ = stage3_loss(fake, gt, composed_x, gate, class_masks=sam_class_masks, class_probs=class_probs)
                outside_lock = torch.mean(torch.abs((1.0 - seam_mask) * (fake - composed_x)))
                g_loss = recon + 0.05 * (system.g_hinge_loss(fg) + system.g_hinge_loss(fs)) + args.outside_lock_weight * outside_lock
            scaler_g.scale(g_loss).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            running_g += g_loss.item()
            pbar.set_postfix(d_loss=f"{running_d / step:.4f}", g_loss=f"{running_g / step:.4f}")

        train_loss = (running_d + running_g) / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start

        val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = 0.0, 0.0, 0.0, 0.0
        if val_loader is not None:
            print("Validating...")
            frozen_stage1.eval()
            frozen_stage2.eval()
            system.generator.eval()
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
            system.train()

        save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3)
        print_epoch_summary(
            epoch, args.epochs, train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3, epoch_time
        )

        payload = {
            "epoch": epoch,
            "generator": system.generator.state_dict(),
            "discriminator": system.discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_hdrvdp2": val_hdrvdp2,
            "val_hdrvdp3": val_hdrvdp3,
            "best_val_psnr": best_psnr,
            "best_val_ssim": best_ssim,
            "best_val_hdrvdp2": best_hdrvdp2,
            "best_val_hdrvdp3": best_hdrvdp3,
            "stage1_ckpt": stage1_path,
            "stage2_ckpt": stage2_path,
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
