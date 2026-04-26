import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from ..dual_decoders.cold_hdr_luminance_diffusion_decoder import Stage1TriEncoderDiffusionSystem
from ..losses.stage_composite_losses import stage1_loss
from ..losses.codebook_losses import kl_codebook_loss
from .common_training import maybe_resume, save_checkpoint
from .data_loader import TriGateHDRDataset


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running = 0.0
    for batch in loader:
        ldr = batch["ldr_image"].to(device)
        hdr = batch["hdr_image"].to(device)
        segmap = batch.get("segmap", ldr).to(device)
        sam_class_masks = batch.get("sam_class_masks", None)
        if sam_class_masks is not None:
            sam_class_masks = sam_class_masks.to(device)
        t = torch.randint(0, 100, (ldr.shape[0],), device=device).long()
        pred, gate, class_probs, aux = model(ldr, t, segmap=segmap)
        loss, _ = stage1_loss(pred, hdr, gate, class_masks=sam_class_masks, class_probs=class_probs)
        loss = loss + 0.01 * kl_codebook_loss(aux["mus"], aux["logvars"])
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
    parser.add_argument("--ldr_dir", type=str, required=False, default="")
    parser.add_argument("--hdr_dir", type=str, required=False, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--sam_mask_dir", type=str, default="")
    parser.add_argument("--max_sam_classes", type=int, default=64)
    parser.add_argument("--continue_train", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Stage1TriEncoderDiffusionSystem().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    start_epoch = 1
    if args.continue_train:
        start_epoch, _, _ = maybe_resume(args.checkpoint_dir, model, optimizer)

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
        train_loss = train_one_epoch(model, loader, optimizer, device) if loader else 0.0
        save_checkpoint(
            args.checkpoint_dir,
            f"epoch_{epoch}",
            {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "train_loss": train_loss},
        )


if __name__ == "__main__":
    main()

