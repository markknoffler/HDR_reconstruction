import csv
import os

import cv2
import numpy as np
import torch

def fhdr_compare_ssim(generated: np.ndarray, real: np.ndarray) -> float:
    """
    FHDR/test.py line 119: compare_ssim(generated, real, multichannel=True).

    Legacy scikit-image: skimage.measure.compare_ssim (multichannel=True on HWC RGB).
    Newer scikit-image: skimage.metrics.structural_similarity with channel_axis=2 only.
    Do not pass multichannel=True to structural_similarity — it treats C as spatial and
    raises win_size errors on normal images.
    """
    try:
        from skimage.measure import compare_ssim

        return float(compare_ssim(generated, real, multichannel=True))
    except ImportError:
        from skimage.metrics import structural_similarity

        kwargs = {"channel_axis": 2}
        min_side = int(min(generated.shape[0], generated.shape[1]))
        if min_side < 7:
            # Same constraint as skimage default win_size=7; only for very thin crops.
            win = min_side if (min_side % 2) else min_side - 1
            kwargs["win_size"] = max(3, win)
        return float(structural_similarity(generated, real, **kwargs))


def mu_tonemap(img):
    """Same as FHDR/util.py (μ-law tonemap, MU=5000)."""
    mu = 5000.0
    return torch.log(1.0 + mu * (img + 1.0) / 2.0) / np.log(1.0 + mu)


def mse_loss(pred, target):
    return torch.mean((pred - target) ** 2)


def _finite_metric(x, default=0.0):
    x = float(x)
    return x if np.isfinite(x) else default


def sanitize_hdr_tensor(t):
    """For saving/visualization only — not used in FHDR-aligned PSNR/SSIM."""
    return torch.nan_to_num(t.float(), nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def compute_psnr_ssim_fhdr(pred, gt):
    """
    PSNR-μ and SSIM exactly as FHDR/test.py (lines 101–119).
    `pred` / `gt`: CHW tensors in the same [-1, 1] HDR space as FHDR ground_truth / output.
    """
    mu_tonemap_gt = mu_tonemap(gt)
    mse = mse_loss(mu_tonemap(pred), mu_tonemap_gt)
    psnr = 10 * np.log10(1 / mse.item())

    generated = (np.transpose(pred.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    real = (np.transpose(gt.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    ssim = fhdr_compare_ssim(generated, real)
    return float(psnr), float(ssim)


def compute_psnr_ssim(pred, gt):
    """Training/validation entry point (FHDR/test.py semantics, no sanitization or fallbacks)."""
    return compute_psnr_ssim_fhdr(pred.detach().float(), gt.detach().float())


def write_hdr(hdr_image, path):
    """Writing HDR image in radiance (.hdr) format (ARThdrNet/utils.py)."""
    norm_image = cv2.cvtColor(hdr_image, cv2.COLOR_BGR2RGB)
    with open(path, "wb") as f:
        vmin, vmax = float(norm_image.min()), float(norm_image.max())
        denom = vmax - vmin
        if denom > 1e-8:
            norm_image = (norm_image - vmin) / denom
        else:
            norm_image = np.clip(norm_image, 0.0, 1.0)
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
    Official HDR-VDP-2 (Q, 0-100) and HDR-VDP-3 quality (Q_JOD) via bundled
    SourceForge releases and Octave. Matches HistoHDR-Net / ICIP evaluation:
    metrics on linear HDR (cd/m^2), not PU21/FovVideoVDP proxies.
    """

    def __init__(self, use_real_hdrvdp=True, peak_luminance=1000.0):
        self._pair_cache = None
        self._pair_cache_key = None
        self._official = None
        self.hdrvdp_available = False

        if use_real_hdrvdp:
            try:
                from ..metrics.hdrvdp_official import OfficialHDRVDPBackend

                self._official = OfficialHDRVDPBackend(peak_luminance=peak_luminance)
                self.hdrvdp_available = self._official.available
                if self.hdrvdp_available:
                    print(
                        f"Official HDR-VDP loaded (Octave: {self._official.octave_executable})"
                    )
                else:
                    print(
                        "WARNING: Official HDR-VDP not available (Octave or third_party missing). "
                        "Metrics will be NaN. Use conda env 'trigate-hdrvdp' or set HDRVDP_OCTAVE_BIN."
                    )
            except Exception as e:
                print(f"WARNING: Official HDR-VDP init failed ({e})")

    def compute_hdrvdp2(self, hdr_pred, hdr_gt):
        q2, q3 = self._compute_official_pair(hdr_pred, hdr_gt)
        return q2

    def compute_hdrvdp3(self, hdr_pred, hdr_gt):
        q2, q3 = self._compute_official_pair(hdr_pred, hdr_gt)
        return q3

    def _compute_official_pair(self, hdr_pred, hdr_gt):
        cache_key = (hdr_pred.data_ptr(), hdr_gt.data_ptr())
        if self._pair_cache_key == cache_key and self._pair_cache is not None:
            return self._pair_cache
        if not self.hdrvdp_available or self._official is None:
            result = (float("nan"), float("nan"))
        else:
            try:
                q2, q3 = self._official.compute_pair(hdr_pred, hdr_gt)
                result = (
                    float(q2) if np.isfinite(q2) else float("nan"),
                    float(q3) if np.isfinite(q3) else float("nan"),
                )
            except Exception as e:
                print(f"WARNING: HDR-VDP computation failed ({e})")
                result = (float("nan"), float("nan"))
        self._pair_cache_key = cache_key
        self._pair_cache = result
        return result


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
                "train_loss": f"{_finite_metric(train_loss):.6f}",
                "val_psnr": f"{_finite_metric(val_psnr):.4f}",
                "val_ssim": f"{_finite_metric(val_ssim):.4f}",
                "val_hdrvdp2": f"{_finite_metric(val_hdrvdp2):.4f}",
                "val_hdrvdp3": f"{_finite_metric(val_hdrvdp3):.4f}",
            }
        )


def maybe_resume(checkpoint_dir, model, optimizer, resume_from: str = ""):
    """
    Load weights/optimizer and return the next epoch to run.

    Uses resume_from if set, else checkpoint_dir/latest.pt.
    If nothing is found, returns start_epoch=1 (fresh run).
    """
    checkpoint_dir = sanitize_data_path(checkpoint_dir)
    if resume_from:
        latest = sanitize_data_path(resume_from)
    else:
        latest = os.path.join(checkpoint_dir, "latest.pt")

    if not os.path.isfile(latest):
        print(
            f"[resume] No checkpoint at {latest!r} — starting from epoch 1.\n"
            f"         Expected finished run: {os.path.join(checkpoint_dir, 'latest.pt')}"
        )
        return 1, 0.0, 0.0, 0.0, 0.0

    ckpt = torch.load(latest, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    last_epoch = int(ckpt.get("epoch", 0))
    start_epoch = last_epoch + 1
    print(
        f"[resume] Loaded {latest}\n"
        f"         last completed epoch={last_epoch} -> continuing at epoch {start_epoch}"
    )
    return (
        start_epoch,
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


def save_latest_checkpoint(checkpoint_dir, payload):
    """Resume checkpoint only (written every epoch; does not create epoch_N.pt)."""
    os.makedirs(checkpoint_dir, exist_ok=True)
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
        default=5,
        help="Save tagged epoch_N.pt every N epochs (metrics/validation run every epoch).",
    )
    parser.add_argument(
        "--final_test_count",
        type=int,
        default=5,
        help="After training finishes, export this many random val LDR->HDR test images.",
    )
    parser.add_argument(
        "--skip_final_test_export",
        action="store_true",
        help="Do not run the post-training random val export.",
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
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Tiny overfit run: 6 train + 4 val images, max_dim=256 (fast sanity check).",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=0,
        help="Cap training images (0=no cap). smoke_test sets 6 if unset.",
    )
    parser.add_argument(
        "--max_val_samples",
        type=int,
        default=0,
        help="Cap validation images (0=no cap). smoke_test sets 4 if unset.",
    )


def sanitize_data_path(path: str) -> str:
    """Strip whitespace/newlines from CLI paths (common copy-paste line-wrap bug)."""
    if not path:
        return path
    cleaned = path.replace("\n", "").replace("\r", "").strip()
    # Shell wrapped "SingleHDR_training_data" -> "SingleHD" + newline + "R_training_data"
    for broken, fixed in (
        ("SingleHD  R_training_data", "SingleHDR_training_data"),
        ("SingleHD R_training_data", "SingleHDR_training_data"),
        ("Sin gleHDR_training_data", "SingleHDR_training_data"),
    ):
        cleaned = cleaned.replace(broken, fixed)
    return os.path.normpath(cleaned)


def default_hrishav_data_paths():
    """Canonical HDR-Real paths for this project (avoids fragile long CLI strings)."""
    root = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR"
    data = os.path.join(root, "SingleHDR_training_data")
    repo = os.path.join(root, "TriGate-HDR")
    return {
        "ldr_dir": os.path.join(data, "HDR-Real", "LDR_in"),
        "hdr_dir": os.path.join(data, "HDR-Real", "HDR_gt"),
        "sam_mask_dir": os.path.join(data, "segmented_masks"),
        "checkpoint_dir": os.path.join(repo, "experiments", "stage1_instruct"),
    }


def apply_smoke_test_args(args):
    if not getattr(args, "smoke_test", False):
        return args
    if args.max_train_samples <= 0:
        args.max_train_samples = 6
    if args.max_val_samples <= 0:
        args.max_val_samples = 4
    if args.max_dim <= 0:
        args.max_dim = 256
    if args.val_export_count > args.max_val_samples:
        args.val_export_count = args.max_val_samples
    return args

