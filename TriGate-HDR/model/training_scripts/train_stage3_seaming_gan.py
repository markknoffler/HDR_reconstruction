import argparse
import torch
import torch.optim as optim

from ..seaming_model.gan_system import SeamingGANSystem
from ..losses.stage_composite_losses import stage3_loss
from .common_training import save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_g", type=float, default=2e-4)
    parser.add_argument("--lr_d", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage3")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = SeamingGANSystem().to(device)
    opt_g = optim.Adam(system.generator.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(system.discriminator.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    loader = []
    for epoch in range(1, args.epochs + 1):
        for batch in loader:
            base_x = batch["base_hdr_x"].to(device)
            gen_clip = batch["generated_clip_hdr"].to(device)
            gate = batch["gate"].to(device)
            gt = batch["hdr_image"].to(device)
            fake, fg, fs = system(base_x, gen_clip, gate)
            rg, rs = system.discriminator(gt, gate)
            d_loss = system.d_hinge_loss(rg, fg.detach()) + system.d_hinge_loss(rs, fs.detach())
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            fake, fg, fs = system(base_x, gen_clip, gate)
            recon, _ = stage3_loss(fake, gt, base_x, gate)
            g_loss = recon + 0.05 * (system.g_hinge_loss(fg) + system.g_hinge_loss(fs))
            opt_g.zero_grad(set_to_none=True)
            g_loss.backward()
            opt_g.step()
        save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", {"epoch": epoch, "generator": system.generator.state_dict(), "discriminator": system.discriminator.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict()})


if __name__ == "__main__":
    main()

