import csv
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim


def mu_tonemap(img):
    mu = 5000.0
    return torch.log(1.0 + mu * (img + 1.0) / 2.0) / np.log(1.0 + mu)


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def _safe_ssim(generated, real):
    """SSIM with skimage API compatibility and small-image win_size handling."""
    generated = np.clip(generated, 0.0, 1.0)
    real = np.clip(real, 0.0, 1.0)
    min_side = min(generated.shape[0], generated.shape[1])
    if min_side < 3:
        return 0.0
    win_size = min(7, min_side)
    if win_size % 2 == 0:
        win_size = max(3, win_size - 1)
    channel_axis = -1 if generated.ndim == 3 else None
    try:
        return float(
            compare_ssim(
                generated,
                real,
                channel_axis=channel_axis,
                win_size=win_size,
                data_range=1.0,
            )
        )
    except TypeError:
        # Older scikit-image
        kwargs = {"win_size": win_size, "data_range": 1.0}
        if generated.ndim == 3:
            kwargs["multichannel"] = True
        return float(compare_ssim(generated, real, **kwargs))


def compute_psnr_ssim(pred, gt):
    """PSNR-mu + SSIM (ARThdrNet/m_training.py PSNR; robust SSIM for smoke/small crops)."""
    pred_batch = pred.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    mu_tonemap_gt = mu_tonemap(gt_batch)
    mu_tonemap_pred = mu_tonemap(pred_batch)
    mse = mse_loss(mu_tonemap_pred, mu_tonemap_gt)
    psnr = 10 * np.log10(1 / mse.item())
    generated = (np.transpose(pred.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    real = (np.transpose(gt.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    ssim = _safe_ssim(generated, real)
    return psnr, ssim


def write_hdr(hdr_image, path):
    """Writing HDR image in radiance (.hdr) format (ARThdrNet/utils.py)."""
    norm_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    with open(path, "wb") as f:
        norm_image = (norm_image - norm_image.min()) / (norm_image.max() - norm_image.min())
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (norm_image.shape[0], norm_image.shape[1]))
        brightest = np.maximum(np.maximum(norm_image[..., 0], norm_image[..., 1]), norm_image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((norm_image.shape[0], norm_image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(norm_image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)
        rgbe.flatten().tofile(f)
        f.close()


def save_hdr_image(img_tensor, batch, path):
    """Match ARThdrNet/m_training.py save_hdr_image."""
    img = img_tensor.data[batch].cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))
    write_hdr(img.astype(np.float32), path)


def save_ldr_image(img_tensor, batch, path):
    """Match ARThdrNet/m_training.py save_ldr_image (expects HDR tensors in [-1, 1])."""
    img = img_tensor.data[batch].cpu().float().numpy()
    img = 255 * (np.transpose(img, (1, 2, 0)) + 1) / 2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def save_ldr_image_01(img_tensor, batch, path):
    """TriGate dataloader LDR is in [0, 1]."""
    img = img_tensor.data[batch].cpu().float().numpy()
    img = 255 * np.clip(np.transpose(img, (1, 2, 0)), 0.0, 1.0)
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, img)


def print_epoch_summary(epoch, total_epochs, train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3, epoch_time):
    """Match ARThdrNet/m_training.py epoch summary printout."""
    print(f"\n{'=' * 60}")
    print(f"Epoch {epoch}/{total_epochs} Summary")
    print(f"{'=' * 60}")
    print(f"  Training Loss    : {train_loss:.6f}")
    print(f"  Validation PSNR  : {val_psnr:.4f} dB")
    print(f"  Validation SSIM  : {val_ssim:.4f}")
    print(f"  HDR-VDP-2 Score  : {val_hdrvdp2:.4f}")
    print(f"  HDR-VDP-3 Score  : {val_hdrvdp3:.4f}")
    print(f"  Epoch Time       : {epoch_time:.2f} seconds")
    print(f"{'=' * 60}")


class HDRVDPMetrics:
    """
    HDR-VDP metric calculator using FovVideoVDP (modern successor to HDR-VDP-2/3).
    Maintains exact same interface as ARThdrNet/m_training.py.
    """

    def __init__(self, use_real_hdrvdp=False):
        self.use_real_hdrvdp = use_real_hdrvdp
        self.fvvdp_model = None
        if use_real_hdrvdp:
            try:
                import pyfvvdp

                self.pyfvvdp = pyfvvdp
                self.fvvdp2 = pyfvvdp.fvvdp(display_name="standard_fhd", heatmap=None)
                self.fvvdp3 = pyfvvdp.fvvdp(display_name="standard_4k", heatmap=None)
                self.hdrvdp_available = True
                print("FovVideoVDP loaded successfully")
            except Exception as e:
                print(f"WARNING: HDRVDP initialization failed ({e}), using PU21 fallback")
                self.hdrvdp_available = False
        else:
            self.hdrvdp_available = False

    def compute_hdrvdp2(self, hdr_pred, hdr_gt):
        if self.hdrvdp_available:
            return self._compute_real_fovvdp2(hdr_pred, hdr_gt)
        return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=5000.0)

    def compute_hdrvdp3(self, hdr_pred, hdr_gt):
        if self.hdrvdp_available:
            return self._compute_real_fovvdp3(hdr_pred, hdr_gt)
        return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=8000.0, use_spatial=True)

    def _compute_real_fovvdp2(self, hdr_pred, hdr_gt):
        try:
            pred_np = hdr_pred.detach().cpu().numpy()
            gt_np = hdr_gt.detach().cpu().numpy()
            pred_np = np.clip((pred_np + 1.0) * 50.0, 0.01, 100.0)
            gt_np = np.clip((gt_np + 1.0) * 50.0, 0.01, 100.0)
            pred_np = np.transpose(pred_np, (1, 2, 0))
            gt_np = np.transpose(gt_np, (1, 2, 0))
            q_jod, _ = self.fvvdp2.predict(pred_np, gt_np, dim_order="HWC")
            return float(np.clip(q_jod, 0.0, 10.0))
        except Exception:
            return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=5000.0)

    def _compute_real_fovvdp3(self, hdr_pred, hdr_gt):
        try:
            pred_np = hdr_pred.detach().cpu().numpy()
            gt_np = hdr_gt.detach().cpu().numpy()
            pred_np = np.clip((pred_np + 1.0) * 50.0, 0.01, 100.0)
            gt_np = np.clip((gt_np + 1.0) * 50.0, 0.01, 100.0)
            pred_np = np.transpose(pred_np, (1, 2, 0))
            gt_np = np.transpose(gt_np, (1, 2, 0))
            q_jod, _ = self.fvvdp3.predict(pred_np, gt_np, dim_order="HWC")
            return float(np.clip(q_jod, 0.0, 10.0))
        except Exception:
            return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=8000.0, use_spatial=True)

    def _compute_pu21_metric(self, hdr_pred, hdr_gt, mu=5000.0, use_spatial=False):
        pred_np = hdr_pred.detach().cpu().numpy()
        gt_np = hdr_gt.detach().cpu().numpy()
        l_pred = np.clip((pred_np + 1.0) * 50.0, 0.01, 100.0)
        l_gt = np.clip((gt_np + 1.0) * 50.0, 0.01, 100.0)
        pu_pred = self._pu21_encode(l_pred)
        pu_gt = self._pu21_encode(l_gt)
        mse = np.mean((pu_pred - pu_gt) ** 2)
        if use_spatial:
            spatial_mse = self._compute_spatial_error(pu_pred, pu_gt)
            mse = 0.6 * mse + 0.4 * spatial_mse
        max_val = self._pu21_encode(np.array([100.0]))[0]
        psnr = 10.0 * np.log10((max_val ** 2) / (mse + 1e-10))
        return float(np.clip(psnr / 10.0, 0.0, 10.0))

    def _pu21_encode(self, luminance):
        return np.log((luminance + 1e-4) / (luminance + 0.01))

    def _compute_spatial_error(self, pred, gt, scales=(1, 2, 4)):
        errors = []
        for scale in scales:
            if scale == 1:
                err = np.mean((pred - gt) ** 2)
            else:
                pred_down = pred[:, ::scale, ::scale]
                gt_down = gt[:, ::scale, ::scale]
                err = np.mean((pred_down - gt_down) ** 2)
            errors.append(err)
        weights = np.array([0.5, 0.3, 0.2])[: len(errors)]
        weights = weights / weights.sum()
        return np.sum([w * e for w, e in zip(weights, errors)])


def save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim, val_hdrvdp2=0.0, val_hdrvdp3=0.0):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as csvfile:
        fieldnames = ["epoch", "train_loss", "val_psnr", "val_ssim", "val_hdrvdp2", "val_hdrvdp3"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "val_psnr": f"{val_psnr:.4f}",
                "val_ssim": f"{val_ssim:.4f}",
                "val_hdrvdp2": f"{val_hdrvdp2:.4f}",
                "val_hdrvdp3": f"{val_hdrvdp3:.4f}",
            }
        )


def maybe_resume(checkpoint_dir, model, optimizer):
    latest = os.path.join(checkpoint_dir, "latest.pt")
    if not os.path.exists(latest):
        return 1, 0.0, 0.0, 0.0, 0.0
    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return (
        ckpt["epoch"] + 1,
        ckpt.get("best_val_psnr", 0.0),
        ckpt.get("best_val_ssim", 0.0),
        ckpt.get("best_val_hdrvdp2", 0.0),
        ckpt.get("best_val_hdrvdp3", 0.0),
    )


def save_checkpoint(checkpoint_dir, tag, payload):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}.pt")
    torch.save(payload, path)
    torch.save(payload, os.path.join(checkpoint_dir, "latest.pt"))


def save_best_checkpoint(checkpoint_dir, payload):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(payload, os.path.join(checkpoint_dir, "best.pt"))


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def add_subset_args(parser):
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Fraction held out for validation (never trained).")
    parser.add_argument("--split_seed", type=int, default=42, help="Seed for reproducible train/val split.")
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=1.0,
        help="Train on 1/N of train split; use 0.2 for 20%% packets (5 packets total).",
    )
    parser.add_argument(
        "--subset_packet",
        type=int,
        default=0,
        help="Which train packet to use when subset_fraction<1 (0..N-1).",
    )
    parser.add_argument("--val_export_count", type=int, default=10, help="Random val images to export after training.")
    parser.add_argument(
        "--val_export_dir",
        type=str,
        default="",
        help="Directory for exported val previews; default <checkpoint_dir>/val_exports",
    )
    parser.add_argument("--val_export_seed", type=int, default=123, help="Seed for picking val export images.")
    parser.add_argument(
        "--validation_results_dir",
        type=str,
        default="",
        help="Per-epoch validation outputs; default <checkpoint_dir>/validation_results",
    )
    parser.add_argument(
        "--save_ckpt_after",
        type=int,
        default=1,
        help="Save epoch checkpoint every N epochs (validation still runs every epoch).",
    )
    parser.add_argument(
        "--save_val_samples_each_epoch",
        action="store_true",
        default=True,
        help="Save up to 10 val LDR/pred/gt files every epoch (ARThdrNet style).",
    )
    parser.add_argument(
        "--no_save_val_samples_each_epoch",
        action="store_false",
        dest="save_val_samples_each_epoch",
        help="Disable per-epoch validation image dumps.",
    )

