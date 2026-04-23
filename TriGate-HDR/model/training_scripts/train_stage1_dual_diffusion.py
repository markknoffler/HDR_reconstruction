import argparse
import torch
import torch.optim as optim

from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import GroundedHDRUNet
from ..losses.stage_composite_losses import stage1_loss
from .common_training import save_checkpoint


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    for batch in loader:
        ldr = batch["ldr_image"].to(device)
        hdr = batch["hdr_image"].to(device)
        mat = batch["mat_feat"].to(device)
        struct = batch["struct_feat"].to(device)
        sem = [x.to(device) for x in batch["sem_feats"]]
        t = torch.randint(0, 100, (ldr.shape[0],), device=device).long()
        pred = model(ldr, t, mat, struct, sem)
        gate = batch["gate"].to(device)
        loss, _ = stage1_loss(pred, hdr, gate)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / max(1, len(loader))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage1")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GroundedHDRUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loader = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, device) if loader else 0.0
        save_checkpoint(args.checkpoint_dir, f"epoch_{epoch}", {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "train_loss": train_loss})


if __name__ == "__main__":
    main()

