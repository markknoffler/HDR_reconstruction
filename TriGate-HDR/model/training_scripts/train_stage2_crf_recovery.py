import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ..decoders.cold_hdr_diffusion_decoder import ColdHDRDiffusion
from ..losses.stage_composite_losses import stage2_loss
from ..losses.radiometric_losses import HybridRadiometricConsistencyLoss
from .common_training import maybe_resume, save_checkpoint
from .data_loader import TriGateHDRDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_stage2")
    parser.add_argument("--ldr_dir", type=str, required=False, default="")
    parser.add_argument("--hdr_dir", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--continue_train", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColdHDRDiffusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    radiometric_loss_fn = HybridRadiometricConsistencyLoss()
    start_epoch = 1
    if args.continue_train:
        start_epoch, _, _ = maybe_resume(args.checkpoint_dir, model, optimizer)
    loader = []
    if args.ldr_dir and args.hdr_dir:
        dataset = TriGateHDRDataset(args.ldr_dir, args.hdr_dir, mode="train")
        train_len = max(1, int(0.8 * len(dataset)))
        val_len = max(0, len(dataset) - train_len)
        train_set, _ = random_split(dataset, [train_len, val_len])
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = 0.0
        model.train()
        for batch in loader:
            ldr = batch["ldr_image"].to(device)
            hdr = batch["hdr_image"].to(device)
            gate = batch["gate"].to(device)
            pred, _ = model(hdr)
            loss, _ = stage2_loss(pred, hdr, ldr, gate, radiometric_loss_fn=radiometric_loss_fn)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / max(1, len(loader))
        save_checkpoint(
            args.checkpoint_dir,
            f"epoch_{epoch}",
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "train_loss": train_loss},
        )


if __name__ == "__main__":
    main()

