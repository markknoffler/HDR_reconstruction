import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from ..seaming_model.gan_system import SeamingGANSystem
from ..losses.stage_composite_losses import stage3_loss
from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import Stage1TriEncoderDiffusionSystem
from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from .common_training import save_checkpoint
from .data_loader import TriGateHDRDataset


def build_composited_input(stage2_hdr, stage1_hdr, gate):
    clip_mask = (1.0 - gate).clamp(0.0, 1.0)
    composed = stage2_hdr * (1.0 - clip_mask) + stage1_hdr * clip_mask
    dilated = F.max_pool2d(clip_mask, kernel_size=17, stride=1, padding=8)
    eroded = -F.max_pool2d(-clip_mask, kernel_size=9, stride=1, padding=4)
    seam_band = (dilated - eroded).clamp(0.0, 1.0)
    seam_band = torch.maximum(seam_band, clip_mask)
    return composed, seam_band, clip_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage3")
    parser.add_argument("--ldr_dir", type=str, required=False, default="")
    parser.add_argument("--hdr_dir", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--stage1_ckpt", type=str, default="")
    parser.add_argument("--stage2_ckpt", type=str, default="")
    parser.add_argument("--sam_mask_dir", type=str, default="")
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument("--outside_lock_weight", type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = SeamingGANSystem().to(device)
    frozen_stage1 = Stage1TriEncoderDiffusionSystem().to(device)
    frozen_stage2 = ColdHDRDiffusion().to(device)
    if args.stage1_ckpt:
        ckpt = torch.load(args.stage1_ckpt, map_location=device)
        if "model" in ckpt:
            frozen_stage1.load_state_dict(ckpt["model"], strict=False)
    if args.stage2_ckpt:
        ckpt = torch.load(args.stage2_ckpt, map_location=device)
        if "model" in ckpt:
            frozen_stage2.load_state_dict(ckpt["model"], strict=False)
    frozen_stage1.eval()
    frozen_stage2.eval()
    for param in frozen_stage1.parameters():
        param.requires_grad = False
    for param in frozen_stage2.parameters():
        param.requires_grad = False
    opt_g = optim.Adam(system.generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(system.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
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
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            ldr = batch["ldr_image"].to(device)
            gate = batch["gate"].to(device)
            gt = batch["hdr_image"].to(device)
            sam_class_masks = batch.get("sam_class_masks", None)
            if sam_class_masks is not None:
                sam_class_masks = sam_class_masks.to(device)
            with torch.no_grad():
                t = torch.randint(0, 100, (ldr.shape[0],), device=device).long()
                gen_clip, _, class_probs, _ = frozen_stage1(ldr, t, segmap=batch.get("segmap", ldr).to(device))
                stage2_hdr = frozen_stage2.restore_hdr(ldr)
                composed_x, seam_mask, _ = build_composited_input(stage2_hdr, gen_clip, gate)
            fake, fg, fs = system(composed_x, gen_clip, seam_mask)
            rg, rs = system.discriminator(gt, seam_mask)
            d_loss = system.d_hinge_loss(rg, fg.detach()) + system.d_hinge_loss(rs, fs.detach())
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            fake, fg, fs = system(composed_x, gen_clip, seam_mask)
            recon, _ = stage3_loss(fake, gt, composed_x, gate, class_masks=sam_class_masks, class_probs=class_probs)
            outside_lock = torch.mean(torch.abs((1.0 - seam_mask) * (fake - composed_x)))
            g_loss = recon + 0.05 * (system.g_hinge_loss(fg) + system.g_hinge_loss(fs)) + args.outside_lock_weight * outside_lock
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
        save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", {"epoch": epoch, "generator": system.generator.state_dict(), "discriminator": system.discriminator.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()})


if __name__ == "__main__":
    main()

