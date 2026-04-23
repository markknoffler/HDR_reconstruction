import argparse
import torch
import torch.optim as optim

from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from ..losses.stage_composite_losses import stage2_loss
from .common_training import save_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage2")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColdHDRDiffusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loader = []
    for epoch in range(1, args.epochs + 1):
        train_loss = 0.0
        model.train()
        for batch in loader:
            ldr = batch["ldr_image"].to(device)
            hdr = batch["hdr_image"].to(device)
            gate = batch["gate"].to(device)
            pred, _ = model(hdr)
            loss, _ = stage2_loss(pred, hdr, ldr, gate)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / max(1, len(loader))
        save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "train_loss": train_loss})


if __name__ == "__main__":
    main()

