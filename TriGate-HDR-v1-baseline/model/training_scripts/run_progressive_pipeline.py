import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import Stage1TriEncoderDiffusionSystem
from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from ..seaming_model.gan_system import SeamingGANSystem
from ..losses.stage_composite_losses import stage3_loss
from .common_training import (
    HDRVDPMetrics,
    compute_psnr_ssim,
    save_metrics_to_csv,
    maybe_resume,
    save_checkpoint,
)
from .data_loader import TriGateHDRDataset


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def build_composited_input(stage2_hdr, stage1_hdr, gate):
    clip_mask = (1.0 - gate).clamp(0.0, 1.0)
    composed = stage2_hdr * (1.0 - clip_mask) + stage1_hdr * clip_mask
    dilated = F.max_pool2d(clip_mask, kernel_size=17, stride=1, padding=8)
    eroded = -F.max_pool2d(-clip_mask, kernel_size=9, stride=1, padding=4)
    seam_band = (dilated - eroded).clamp(0.0, 1.0)
    seam_band = torch.maximum(seam_band, clip_mask)
    return composed, seam_band


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_pipeline")
    parser.add_argument("--save_ckpt_after", type=int, default=2)
    parser.add_argument("--csv_file", type=str, default="training_metrics.csv")
    parser.add_argument("--ldr_dir", type=str, required=False, default="")
    parser.add_argument("--hdr_dir", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sam_mask_dir", type=str, default="")
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument("--outside_lock_weight", type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = Stage1TriEncoderDiffusionSystem().to(device)
    stage2 = ColdHDRDiffusion().to(device)
    stage3 = SeamingGANSystem().to(device)
    hdrvdp_calculator = HDRVDPMetrics()
    freeze_module(stage1)
    freeze_module(stage2)
    optimizer = torch.optim.Adam(stage3.generator.parameters(), lr=2e-4)
    start_epoch, best_psnr, best_ssim = (1, 0.0, 0.0)
    if args.continue_train:
        start_epoch, best_psnr, best_ssim = maybe_resume(args.checkpoint_dir, stage3.generator, optimizer)

    loader = []
    if args.ldr_dir and args.hdr_dir:
        dataset = TriGateHDRDataset(
            args.ldr_dir,
            args.hdr_dir,
            mode="train",
            sam_mask_dir=args.sam_mask_dir,
            max_sam_classes=args.max_sam_classes,
        )
        train_len = max(1, int(0.8 * len(dataset)))
        val_len = max(0, len(dataset) - train_len)
        train_set, _ = random_split(dataset, [train_len, val_len])
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for epoch in range(start_epoch, args.epochs + 1):
        stage3.train()
        running = 0.0
        total_psnr, total_ssim, total_hdrvdp2, total_hdrvdp3 = 0.0, 0.0, 0.0, 0.0
        num_samples = 0
        for batch in loader:
            ldr = batch["ldr_image"].to(device)
            hdr_gt = batch["hdr_image"].to(device)
            gate = batch["gate"].to(device)
            sam_class_masks = batch.get("sam_class_masks", None)
            if sam_class_masks is not None:
                sam_class_masks = sam_class_masks.to(device)
            t = torch.randint(0, 100, (ldr.shape[0],), device=device).long()
            gen_clip, _, class_probs, _ = stage1(ldr, t, segmap=batch.get("segmap", ldr).to(device))
            stage2_hdr = stage2.restore_hdr(ldr, gate=gate)
            composed_x, seam_mask = build_composited_input(stage2_hdr, gen_clip, gate)
            fake = stage3.generator(composed_x, gen_clip, seam_mask)
            recon_loss, _ = stage3_loss(fake, hdr_gt, composed_x, gate, class_masks=sam_class_masks, class_probs=class_probs)
            outside_lock = torch.mean(torch.abs((1.0 - seam_mask) * (fake - composed_x)))
            loss = recon_loss + args.outside_lock_weight * outside_lock
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
            for i in range(fake.shape[0]):
                psnr, ssim = compute_psnr_ssim(fake[i], hdr_gt[i])
                hdrvdp2 = hdrvdp_calculator.compute_hdrvdp2(fake[i], hdr_gt[i])
                hdrvdp3 = hdrvdp_calculator.compute_hdrvdp3(fake[i], hdr_gt[i])
                total_psnr += psnr
                total_ssim += ssim
                total_hdrvdp2 += hdrvdp2
                total_hdrvdp3 += hdrvdp3
                num_samples += 1

        avg_loss = running / max(1, len(loader))
        val_psnr = total_psnr / num_samples if num_samples > 0 else 0
        val_ssim = total_ssim / num_samples if num_samples > 0 else 0
        val_hdrvdp2 = total_hdrvdp2 / num_samples if num_samples > 0 else 0
        val_hdrvdp3 = total_hdrvdp3 / num_samples if num_samples > 0 else 0
        save_metrics_to_csv(args.csv_file, epoch, avg_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3)
        client_state = {
            "epoch": epoch,
            "best_val_psnr": best_psnr,
            "best_val_ssim": best_ssim,
            "avg_train_loss": avg_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_hdrvdp2": val_hdrvdp2,
            "val_hdrvdp3": val_hdrvdp3,
        }
        if val_psnr > best_psnr:
            best_psnr, best_ssim = val_psnr, val_ssim
            save_checkpoint(args.checkpoint_dir, f"best_epoch_{epoch}", {"epoch": epoch, "model": stage3.generator.state_dict(), "optimizer": optimizer.state_dict(), **client_state})
        if epoch % args.save_ckpt_after == 0:
            save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", {"epoch": epoch, "model": stage3.generator.state_dict(), "optimizer": optimizer.state_dict(), **client_state})


if __name__ == "__main__":
    main()

