import argparse
import os

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from ..losses.stage_composite_losses import stage2_loss
from ..losses.radiometric_losses import HybridRadiometricConsistencyLoss
from .common_training import add_subset_args, load_checkpoint, maybe_resume, save_best_checkpoint, save_checkpoint, save_metrics_to_csv
from .dataset_splits import build_dataloaders
from .val_export import export_stage2_samples, pick_val_export_indices, validate_stage2


def main():
    parser = argparse.ArgumentParser(description="Train Stage-2 cold HDR diffusion.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage2")
    parser.add_argument("--ldr_dir", type=str, required=False, default="")
    parser.add_argument("--hdr_dir", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--continue_train", action="store_true")
    add_subset_args(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColdHDRDiffusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    radiometric_loss_fn = HybridRadiometricConsistencyLoss()

    start_epoch = 1
    best_psnr, best_ssim = 0.0, 0.0
    if args.continue_train:
        start_epoch, best_psnr, best_ssim = maybe_resume(args.checkpoint_dir, model, optimizer)

    train_loader, val_loader, full_dataset, val_indices = ([], None, None, [])
    if args.ldr_dir and args.hdr_dir:
        train_loader, val_loader, full_dataset, val_indices = build_dataloaders(
            args.ldr_dir,
            args.hdr_dir,
            args.batch_size,
            max_dim=args.max_dim,
            val_ratio=args.val_ratio,
            split_seed=args.split_seed,
            subset_fraction=args.subset_fraction,
            subset_packet=args.subset_packet,
            checkpoint_dir=args.checkpoint_dir,
        )
        print(
            f"[Stage2] train={len(train_loader.dataset)} val={len(val_loader.dataset) if val_loader else 0} "
            f"packet={args.subset_packet} fraction={args.subset_fraction}"
        )

    csv_path = os.path.join(args.checkpoint_dir, "training_metrics.csv")
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Stage2 Epoch {epoch}/{args.epochs}", leave=True)
        for step, batch in enumerate(pbar, start=1):
            ldr = batch["ldr_image"].to(device)
            hdr = batch["hdr_image"].to(device)
            gate = batch["gate"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=args.amp and device.type == "cuda"):
                pred, _ = model(hdr)
                loss, _ = stage2_loss(pred, hdr, ldr, gate, radiometric_loss_fn=radiometric_loss_fn)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{train_loss / step:.4f}")

        train_loss = train_loss / max(1, len(train_loader))
        val_psnr, val_ssim = (0.0, 0.0)
        if val_loader is not None:
            val_psnr, val_ssim = validate_stage2(model, val_loader, device, amp=args.amp)

        save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim)
        payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "best_val_psnr": best_psnr,
            "best_val_ssim": best_ssim,
        }
        save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", payload)

        if val_psnr > best_psnr:
            best_psnr, best_ssim = val_psnr, val_ssim
            payload["best_val_psnr"] = best_psnr
            payload["best_val_ssim"] = best_ssim
            save_best_checkpoint(args.checkpoint_dir, payload)
            print(f"[Stage2] new best epoch={epoch} psnr={best_psnr:.4f} ssim={best_ssim:.4f}")

        print(f"[Stage2] epoch={epoch} train_loss={train_loss:.6f} val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}")

    if full_dataset is not None and val_indices:
        best_path = os.path.join(args.checkpoint_dir, "best.pt")
        ckpt_path = best_path if os.path.isfile(best_path) else os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.isfile(ckpt_path):
            ckpt = load_checkpoint(ckpt_path, device)
            model.load_state_dict(ckpt["model"], strict=False)
            export_dir = args.val_export_dir or os.path.join(args.checkpoint_dir, "val_exports")
            export_idx = pick_val_export_indices(val_indices, args.val_export_count, args.val_export_seed)
            export_stage2_samples(model, full_dataset, export_idx, export_dir, device, amp=args.amp)
            print(f"[Stage2] exported {len(export_idx)} val images to {export_dir} using {ckpt_path}")


if __name__ == "__main__":
    main()
