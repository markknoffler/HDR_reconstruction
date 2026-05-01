import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch


def read_ldr_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read LDR image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
    return img


def build_semantic_map_from_sam_masks(mask_list, h: int, w: int):
    semantic_map = np.zeros((h, w), dtype=np.uint16)
    sorted_masks = sorted(mask_list, key=lambda x: x.get("area", 0), reverse=True)
    metadata = []
    class_id = 1
    for m in sorted_masks:
        seg = m["segmentation"].astype(bool)
        semantic_map[seg] = class_id
        metadata.append(
            {
                "class_id": int(class_id),
                "area": int(m.get("area", int(seg.sum()))),
                "pred_iou": float(m.get("predicted_iou", 0.0)),
                "stability_score": float(m.get("stability_score", 0.0)),
                "bbox_xywh": [int(v) for v in m.get("bbox", [0, 0, 0, 0])],
            }
        )
        class_id += 1
    return semantic_map, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldr_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="vit_h")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--pred_iou_thresh", type=float, default=0.86)
    parser.add_argument("--stability_score_thresh", type=float, default=0.92)
    args = parser.parse_args()

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    ldr_dir = Path(args.ldr_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata").mkdir(parents=True, exist_ok=True)

    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )

    ldr_files = sorted([p for p in ldr_dir.iterdir() if p.suffix.lower() in [".png", ".jpg", ".jpeg"]])
    if not ldr_files:
        print(f"No LDR files found in: {ldr_dir}")
        return

    for idx, ldr_path in enumerate(ldr_files, start=1):
        rgb8 = read_ldr_image(ldr_path)
        masks = mask_generator.generate(rgb8)

        semantic_map, metadata = build_semantic_map_from_sam_masks(masks, rgb8.shape[0], rgb8.shape[1])
        stem = ldr_path.stem
        np.savez_compressed(
            out_dir / f"{stem}.npz",
            semantic_map=semantic_map.astype(np.uint16),
            height=np.int32(rgb8.shape[0]),
            width=np.int32(rgb8.shape[1]),
            num_classes=np.int32(int(semantic_map.max())),
        )
        with open(out_dir / "metadata" / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump({"file": ldr_path.name, "num_classes": int(semantic_map.max()), "segments": metadata}, f, indent=2)
        print(f"[{idx}/{len(ldr_files)}] Saved SAM masks for {ldr_path.name}")

    print(f"Done. SAM masks saved at: {out_dir}")


if __name__ == "__main__":
    main()

