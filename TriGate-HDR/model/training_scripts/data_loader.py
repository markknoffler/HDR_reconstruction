import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class TriGateHDRDataset(Dataset):
    def __init__(self, ldr_dir, hdr_dir, mode="train", sam_mask_dir="", max_sam_classes=64):
        self.ldr_dir = ldr_dir
        self.hdr_dir = hdr_dir
        self.mode = mode
        self.sam_mask_dir = sam_mask_dir
        self.max_sam_classes = max_sam_classes
        ldr_files = sorted([f for f in os.listdir(ldr_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        self.pairs = []
        for ldr_file in ldr_files:
            stem = os.path.splitext(ldr_file)[0]
            candidates = [f"{stem}.hdr", f"{stem}.exr", f"{stem}.npy"]
            for cand in candidates:
                if os.path.exists(os.path.join(hdr_dir, cand)):
                    self.pairs.append((ldr_file, cand))
                    break

    def __len__(self):
        return len(self.pairs)

    def _load_hdr(self, path):
        if path.endswith(".npy"):
            return np.load(path).astype(np.float32)
        hdr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if hdr is None:
            raise ValueError(f"Failed to load HDR image: {path}")
        return cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB).astype(np.float32)

    def __getitem__(self, idx):
        ldr_name, hdr_name = self.pairs[idx]
        ldr_path = os.path.join(self.ldr_dir, ldr_name)
        hdr_path = os.path.join(self.hdr_dir, hdr_name)

        ldr = cv2.imread(ldr_path)
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        hdr = self._load_hdr(hdr_path)
        hdr = np.clip(hdr, 0.0, None)
        hdr_max = float(hdr.max())
        if hdr_max > 0:
            hdr = hdr / hdr_max

        ldr_t = torch.from_numpy(ldr).permute(2, 0, 1).float()
        hdr_t = torch.from_numpy(2.0 * hdr - 1.0).permute(2, 0, 1).float()
        gate = (ldr_t.max(dim=0, keepdim=True).values < 0.98).float()
        sam_class_masks = torch.zeros((self.max_sam_classes, ldr_t.shape[1], ldr_t.shape[2]), dtype=torch.float32)
        segmap_t = ldr_t
        if self.sam_mask_dir:
            stem = os.path.splitext(ldr_name)[0]
            sam_path = os.path.join(self.sam_mask_dir, f"{stem}.npz")
            if os.path.exists(sam_path):
                sam_npz = np.load(sam_path)
                sem_map = torch.from_numpy(sam_npz["semantic_map"].astype(np.int64))
                if sem_map.shape != (ldr_t.shape[1], ldr_t.shape[2]):
                    sem_map = F.interpolate(
                        sem_map[None, None].float(),
                        size=(ldr_t.shape[1], ldr_t.shape[2]),
                        mode="nearest",
                    )[0, 0].long()
                num_cls = min(int(sem_map.max().item()), self.max_sam_classes)
                for cid in range(1, num_cls + 1):
                    sam_class_masks[cid - 1] = (sem_map == cid).float()
                sem_norm = sem_map.float() / max(1.0, float(num_cls))
                dx = torch.abs(sem_norm[:, 1:] - sem_norm[:, :-1])
                dy = torch.abs(sem_norm[1:, :] - sem_norm[:-1, :])
                edge = torch.zeros_like(sem_norm)
                edge[:, 1:] += dx
                edge[1:, :] += dy
                edge = torch.clamp(edge, 0.0, 1.0)
                cls_presence = torch.clamp(sam_class_masks[:num_cls].sum(dim=0), 0.0, 1.0)
                segmap_t = torch.stack([sem_norm, edge, cls_presence], dim=0)
            else:
                sam_class_masks[0] = 1.0
        else:
            sam_class_masks[0] = 1.0

        return {
            "ldr_image": ldr_t,
            "hdr_image": hdr_t,
            "segmap": segmap_t,
            "gate": gate,
            "sam_class_masks": sam_class_masks,
            "ldr_path": ldr_name,
            "hdr_path": hdr_name,
        }

