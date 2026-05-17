import argparse
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..seaming_model.gan_system import SeamingGANSystem
from ..losses.stage_composite_losses import stage3_loss
from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import Stage1TriEncoderDiffusionSystem
from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from .common_training import add_subset_args, load_checkpoint, save_best_checkpoint, save_checkpoint, save_metrics_to_csv
from .dataset_splits import build_dataloaders
from .val_export import export_stage3_samples, pick_val_export_indices, validate_stage3


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    start_epoch = 1
    best_psnr, best_ssim = 0.0, 0.0
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

    train_loader, val_loader, full_dataset, val_indices = ([], None, None, [])
    if args.ldr_dir and args.hdr_dir:
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
        )
        print(
            f"[Stage3] train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0} "
            f"packet={args.subset_packet} fraction={args.subset_fraction}"
        )

    csv_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")
    for epoch in range(start_epoch, args.epochs + 1):
        system.train()
        running_d, running_g = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Stage3 Epoch {epoch}/{args.epochs}", leave=True)
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
                    stage2_hdr = frozen_stage2.restore_hdr(ldr)
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
        val_psnr, val_ssim = (0.0, 0.0)
        if val_loader is not None:
            val_psnr, val_ssim = validate_stage3(
                frozen_stage1, frozen_stage2, system.generator, val_loader, device, amp=args.amp
            )

        save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim)
        payload = {
            "epoch": epoch,
            "generator": system.generator.state_dict(),
            "discriminator": system.discriminator.state_dict(),
            "opt_g": opt_g.state_dict(),
            "opt_d": opt_d.state_dict(),
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_val_psnr": best_psnr,
            "best_val_ssim": best_ssim,
            "stage1_ckpt": stage1_path,
            "stage2_ckpt": stage2_path,
        }
        save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", payload)

        if val_psnr > best_psnr:
            best_psnr, best_ssim = val_psnr, val_ssim
            payload["best_val_psnr"] = best_psnr
            payload["best_val_ssim"] = best_ssim
            save_best_checkpoint(args.checkpoint_dir, payload)
            print(f"[Stage3] new best epoch={epoch} psnr={best_psnr:.4f} ssim={best_ssim:.4f}")

        print(
            f"[Stage3] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}"
        )

    if full_dataset is not None and val_indices:
        best_path = os.path.join(args.checkpoint_dir, "best.pt")
        ckpt_path = best_path if os.path.isfile(best_path) else os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.isfile(ckpt_path):
            ckpt = load_checkpoint(ckpt_path, device)
            system.generator.load_state_dict(ckpt["generator"], strict=False)
            export_dir = args.val_export_dir or os.path.join(args.checkpoint_dir, "val_exports")
            export_idx = pick_val_export_indices(val_indices, args.val_export_count, args.val_export_seed)
            export_stage3_samples(
                frozen_stage1, frozen_stage2, system.generator, full_dataset, export_idx, export_dir, device, amp=args.amp
            )
            print(f"[Stage3] exported {len(export_idx)} full-pipeline val images to {export_dir} using {ckpt_path}")


if __name__ == "__main__":
    main()
