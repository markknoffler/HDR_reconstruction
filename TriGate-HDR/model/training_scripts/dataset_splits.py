"""Fixed train/val splits and optional 20% train packets for smoke tests."""

import json
import os
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Subset

from .data_loader import TriGateHDRDataset


def _packet_count(subset_fraction: float) -> int:
    if subset_fraction <= 0 or subset_fraction > 1.0:
        raise ValueError("subset_fraction must be in (0, 1]")
    # e.g. fraction=0.2 -> 5 packets of ~20% each
    n = max(1, int(round(1.0 / subset_fraction)))
    return n


def compute_split_indices(
    num_samples: int,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    subset_fraction: float = 1.0,
    subset_packet: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Returns (train_indices, val_indices, train_packet_indices).

    - val_indices: held-out validation (never trained on).
    - train_packet_indices: subset of train_indices for this packet (e.g. 20% of train).
    """
    if num_samples < 2:
        train_idx = list(range(num_samples))
        return train_idx, [], train_idx

    rng = np.random.default_rng(split_seed)
    all_idx = np.arange(num_samples)
    rng.shuffle(all_idx)

    val_count = max(1, int(round(num_samples * val_ratio)))
    val_count = min(val_count, num_samples - 1)
    val_indices = sorted(all_idx[:val_count].tolist())
    train_indices = sorted(all_idx[val_count:].tolist())

    if subset_fraction >= 1.0:
        return train_indices, val_indices, train_indices

    n_packets = _packet_count(subset_fraction)
    if subset_packet < 0 or subset_packet >= n_packets:
        raise ValueError(f"subset_packet must be in [0, {n_packets - 1}] for fraction={subset_fraction}")

    train_arr = np.array(train_indices)
    packet_size = max(1, int(len(train_arr) // n_packets))
    start = subset_packet * packet_size
    end = len(train_arr) if subset_packet == n_packets - 1 else start + packet_size
    packet_indices = train_arr[start:end].tolist()
    return train_indices, val_indices, packet_indices


def save_split_manifest(checkpoint_dir: str, manifest: dict) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "split_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def build_dataloaders(
    ldr_dir: str,
    hdr_dir: str,
    batch_size: int,
    sam_mask_dir: str = "",
    max_sam_classes: int = 64,
    max_dim: int = 0,
    val_ratio: float = 0.2,
    split_seed: int = 42,
    subset_fraction: float = 1.0,
    subset_packet: int = 0,
    checkpoint_dir: str = "",
    num_workers: int = 0,
    max_train_samples: int = 0,
    max_val_samples: int = 0,
):
    full_dataset = TriGateHDRDataset(
        ldr_dir,
        hdr_dir,
        mode="train",
        sam_mask_dir=sam_mask_dir,
        max_sam_classes=max_sam_classes,
        max_dim=max_dim,
    )
    train_idx, val_idx, packet_idx = compute_split_indices(
        len(full_dataset),
        val_ratio=val_ratio,
        split_seed=split_seed,
        subset_fraction=subset_fraction,
        subset_packet=subset_packet,
    )

    if max_train_samples > 0:
        packet_idx = packet_idx[:max_train_samples]
    if max_val_samples > 0:
        val_idx = val_idx[:max_val_samples]

    train_set = Subset(full_dataset, packet_idx)
    val_set = Subset(full_dataset, val_idx) if val_idx else None

    if checkpoint_dir:
        save_split_manifest(
            checkpoint_dir,
            {
                "num_samples": len(full_dataset),
                "train_indices_all": train_idx,
                "train_indices_packet": packet_idx,
                "val_indices": val_idx,
                "val_ratio": val_ratio,
                "split_seed": split_seed,
                "subset_fraction": subset_fraction,
                "subset_packet": subset_packet,
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
            },
        )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(val_set, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        if val_set is not None
        else None
    )
    return train_loader, val_loader, full_dataset, val_idx
