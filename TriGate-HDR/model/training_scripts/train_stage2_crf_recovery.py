import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm

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
    parser.add_argument("--max_dim", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--continue_train", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ColdHDRDiffusion().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    radiometric_loss_fn = HybridRadiometricConsistencyLoss()
    start_epoch = 1
    if args.continue_train:
        start_epoch, _, _ = maybe_resume(args.checkpoint_dir, model, optimizer)
    loader = []
    if args.ldr_dir and args.hdr_dir:
        dataset = TriGateHDRDataset(args.ldr_dir, args.hdr_dir, mode="train", max_dim=args.max_dim)
        train_len = max(1, int(0.8 * len(dataset)))
        val_len = max(0, len(dataset) - train_len)
        train_set, _ = random_split(dataset, [train_len, val_len])
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = 0.0
        model.train()
        pbar = tqdm(loader, desc=f"Stage2 Epoch {epoch}/{args.epochs}", leave=True)
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
        train_loss = train_loss / max(1, len(loader))
        print(f"[Stage2] epoch={epoch} train_loss={train_loss:.6f}")
        save_checkpoint(
            args.checkpoint_dir,
            f"epoch_{epoch}",
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "train_loss": train_loss},
        )


if __name__ == "__main__":
    main()

