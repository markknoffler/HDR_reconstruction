import csv
import os
import numpy as np
import torch
from skimage.metrics import structural_similarity as compare_ssim


def mu_tonemap(img):
    mu = 5000.0
    return torch.log(1.0 + mu * (img + 1.0) / 2.0) / np.log(1.0 + mu)


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def compute_psnr_ssim(pred, gt, avg_psnr=None, avg_ssim=None):
    pred_batch = pred.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    mu_tonemap_gt = mu_tonemap(gt_batch)
    mu_tonemap_pred = mu_tonemap(pred_batch)
    mse = mse_loss(mu_tonemap_pred, mu_tonemap_gt)
    psnr = 10 * np.log10(1 / mse.item())
    generated = (np.transpose(pred.detach().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    real = (np.transpose(gt.detach().cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    ssim = compare_ssim(generated, real, multichannel=True)
    return psnr, ssim


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
        return 1, 0.0, 0.0
    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"] + 1, ckpt.get("best_val_psnr", 0.0), ckpt.get("best_val_ssim", 0.0)


def save_checkpoint(checkpoint_dir, tag, payload):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{tag}.pt")
    torch.save(payload, path)
    torch.save(payload, os.path.join(checkpoint_dir, "latest.pt"))

