import argparse
import torch

from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import GroundedHDRUNet
from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from ..seaming_model.gan_system import SeamingGANSystem
from .common_training import (
    HDRVDPMetrics,
    compute_psnr_ssim,
    save_metrics_to_csv,
    maybe_resume,
    save_checkpoint,
)


def freeze_module(module):
    for p in module.parameters():
        p.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_pipeline")
    parser.add_argument("--save_ckpt_after", type=int, default=2)
    parser.add_argument("--csv_file", type=str, default="training_metrics.csv")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage1 = GroundedHDRUNet().to(device)
    stage2 = ColdHDRDiffusion().to(device)
    stage3 = SeamingGANSystem().to(device)
    hdrvdp_calculator = HDRVDPMetrics(use_real_hdrvdp=False)
    freeze_module(stage1)
    freeze_module(stage2)
    optimizer = torch.optim.Adam(stage3.generator.parameters(), lr=2e-4)
    start_epoch, best_psnr, best_ssim = (1, 0.0, 0.0)
    if args.continue_train:
        start_epoch, best_psnr, best_ssim = maybe_resume(args.checkpoint_dir, stage3.generator, optimizer)

    loader = []
    for epoch in range(start_epoch, args.epochs + 1):
        stage3.train()
        running = 0.0
        total_psnr, total_ssim, total_hdrvdp2, total_hdrvdp3, n = 0.0, 0.0, 0.0, 0.0, 0
        for batch in loader:
            ldr = batch["ldr_image"].to(device)
            hdr_gt = batch["hdr_image"].to(device)
            mat = batch["mat_feat"].to(device)
            struct = batch["struct_feat"].to(device)
            sem = [x.to(device) for x in batch["sem_feats"]]
            gate = batch["gate"].to(device)
            t = torch.randint(0, 100, (ldr.shape[0],), device=device).long()
            gen_clip = stage1(ldr, t, mat, struct, sem)
            base_x = stage2.restore_hdr(ldr)
            fake = stage3.generator(base_x, gen_clip, gate)
            loss = torch.mean((fake - hdr_gt) ** 2)
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
                n += 1

        avg_loss = running / max(1, len(loader))
        val_psnr = total_psnr / max(1, n)
        val_ssim = total_ssim / max(1, n)
        val_hdrvdp2 = total_hdrvdp2 / max(1, n)
        val_hdrvdp3 = total_hdrvdp3 / max(1, n)
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

