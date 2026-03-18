import os
import csv
import cv2
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from PIL import Image
import imageio.v3 as iio
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
from datetime import datetime
import glob
from utils import make_required_directories, mu_tonemap, save_hdr_image, save_ldr_image

mse_loss = nn.MSELoss()


from model import HistoHDRNet
from losses import HistoHDRNetLoss

#import matlab.engine
#_matlab_eng = None

def init_matlab_engine():
    global _matlab_eng
    if _matlab_eng is None:
        _matlab_eng = matlab.engine.start_matlab()
        _matlab_eng.addpath(r'./hdrvdp-2.2.1/', nargout=0)  # UPDATE YOUR PATH!
        print("✓ HDR-VDP-2 MATLAB engine initialized")
    return _matlab_eng

def init_hdrvdp2():
    """Initialize MATLAB engine once for HDR-VDP-2"""
    init_matlab_engine()


LDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"  # Directory containing LDR images (.jpg)
HDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_SCRIPT_DIR, "checkpoints")
GENERATED_DIR = "./generated_images"
CSV_LOG_FILE = "./training_log.csv"
BATCH_SIZE = 2
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
IMAGE_SIZE = 512
ACCUMULATION_STEPS = 5
DEVICE = "cuda:0"

# Ensure output dirs exist (at import and again at train start)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)


class HDRDataset(Dataset):
    def __init__(self, ldr_dir, hdr_dir, image_size=512, mode='train'):
        self.ldr_dir = ldr_dir
        self.hdr_dir = hdr_dir
        self.image_size = image_size
        self.mode = mode
        
        self.ldr_files = sorted(glob.glob(os.path.join(ldr_dir, "*.jpg")))
        if len(self.ldr_files) == 0:
            self.ldr_files = sorted(glob.glob(os.path.join(ldr_dir, "*.png")))
        
        self.hdr_files = sorted(glob.glob(os.path.join(hdr_dir, "*.hdr")))
        
        if len(self.ldr_files) != len(self.hdr_files):
            print(f"Warning: LDR files ({len(self.ldr_files)}) != HDR files ({len(self.hdr_files)})")
            min_len = min(len(self.ldr_files), len(self.hdr_files))
            self.ldr_files = self.ldr_files[:min_len]
            self.hdr_files = self.hdr_files[:min_len]
        
        print(f"{mode} dataset: {len(self.ldr_files)} image pairs")
    
    def load_ldr(self, path):
        """Load LDR image. Returns [0,1] for histogram_equalization; caller converts to [-1,1]."""
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        return img

    def load_hdr(self, path):
        """FHDR-style HDR loading: no clipping to [0,1], preserve full dynamic range."""
        hdr = iio.imread(path)
        hdr = cv2.resize(hdr, (self.image_size, self.image_size))
        hdr = np.clip(hdr, 0, None)  # Only clip negatives; preserve HDR range
        return hdr.astype(np.float32)
    
    def histogram_equalization(self, img):
        img_uint8 = (img * 255).astype(np.uint8)
        img_yuv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        img_eq = img_eq.astype(np.float32) / 255.0
        return img_eq
    
    def __len__(self):
        return len(self.ldr_files)
    
    def __getitem__(self, idx):
        ldr_path = self.ldr_files[idx]
        hdr_path = self.hdr_files[idx]
        
        ldr_gt = self.load_ldr(ldr_path)
        ldr_his = self.histogram_equalization(ldr_gt)
        hdr_gt = self.load_hdr(hdr_path)
        
        # FHDR-style LDR: Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) -> [-1, 1]
        ldr_gt = 2.0 * ldr_gt - 1.0
        ldr_his = 2.0 * ldr_his - 1.0

        # FHDR-style HDR: Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) -> (x - 0.5) / 0.5
        hdr_gt = (hdr_gt - 0.5) / 0.5

        ldr_gt = torch.from_numpy(ldr_gt.astype(np.float32)).permute(2, 0, 1)
        ldr_his = torch.from_numpy(ldr_his.astype(np.float32)).permute(2, 0, 1)
        hdr_gt = torch.from_numpy(hdr_gt.astype(np.float32)).permute(2, 0, 1)

        return ldr_gt, ldr_his, hdr_gt, os.path.basename(ldr_path)


class HDRVDPMetrics:
    """
    HDR-VDP metric calculator (same as m_training.py / enhanced_model).
    Uses FovVideoVDP if pyfvvdp available, else PU21-based proxy.
    """
    def __init__(self, use_real_hdrvdp=False):
        self.use_real_hdrvdp = use_real_hdrvdp
        self.fvvdp_model = None
        if use_real_hdrvdp:
            try:
                import pyfvvdp
                self.fvvdp2 = pyfvvdp.fvvdp(display_name='standard_fhd', heatmap=None)
                self.fvvdp3 = pyfvvdp.fvvdp(display_name='standard_4k', heatmap=None)
                self.hdrvdp_available = True
                print("✓ FovVideoVDP loaded (HDR-VDP-2/3)")
            except Exception as e:
                self.hdrvdp_available = False
                print(f"⚠ pyfvvdp not available ({e}), using PU21 proxy")
        else:
            self.hdrvdp_available = False
            print("Using PU21-based HDR-VDP proxy metrics")

    def compute_hdrvdp2(self, hdr_pred, hdr_gt):
        """HDR-VDP-2 style score; inputs (C,H,W) in [-1,1]."""
        if self.hdrvdp_available:
            return self._compute_real_fovvdp2(hdr_pred, hdr_gt)
        return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=5000.0)

    def compute_hdrvdp3(self, hdr_pred, hdr_gt):
        """HDR-VDP-3 style score; inputs (C,H,W) in [-1,1]."""
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
            Q_JOD, _ = self.fvvdp2.predict(pred_np, gt_np, dim_order='HWC')
            return float(np.clip(Q_JOD, 0.0, 10.0))
        except Exception as e:
            return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=5000.0)

    def _compute_real_fovvdp3(self, hdr_pred, hdr_gt):
        try:
            pred_np = hdr_pred.detach().cpu().numpy()
            gt_np = hdr_gt.detach().cpu().numpy()
            pred_np = np.clip((pred_np + 1.0) * 50.0, 0.01, 100.0)
            gt_np = np.clip((gt_np + 1.0) * 50.0, 0.01, 100.0)
            pred_np = np.transpose(pred_np, (1, 2, 0))
            gt_np = np.transpose(gt_np, (1, 2, 0))
            Q_JOD, _ = self.fvvdp3.predict(pred_np, gt_np, dim_order='HWC')
            return float(np.clip(Q_JOD, 0.0, 10.0))
        except Exception as e:
            return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=8000.0, use_spatial=True)

    def _compute_pu21_metric(self, hdr_pred, hdr_gt, mu=5000.0, use_spatial=False):
        pred_np = hdr_pred.detach().cpu().numpy()
        gt_np = hdr_gt.detach().cpu().numpy()
        L_pred = np.clip((pred_np + 1.0) * 50.0, 0.01, 100.0)
        L_gt = np.clip((gt_np + 1.0) * 50.0, 0.01, 100.0)
        pu_pred = np.log((L_pred + 1e-4) / (L_pred + 0.01))
        pu_gt = np.log((L_gt + 1e-4) / (L_gt + 0.01))
        mse = np.mean((pu_pred - pu_gt) ** 2)
        if use_spatial:
            spatial_mse = self._compute_spatial_error(pu_pred, pu_gt)
            mse = 0.6 * mse + 0.4 * spatial_mse
        max_val = np.log((100.0 + 1e-4) / (100.0 + 0.01))
        psnr = 10.0 * np.log10((max_val ** 2) / (mse + 1e-10))
        return float(np.clip(psnr / 10.0, 0.0, 10.0))

    def _compute_spatial_error(self, pred, gt, scales=[1, 2, 4]):
        errors = []
        for scale in scales:
            if scale == 1:
                err = np.mean((pred - gt) ** 2)
            else:
                pred_down = pred[:, ::scale, ::scale]
                gt_down = gt[:, ::scale, ::scale]
                err = np.mean((pred_down - gt_down) ** 2)
            errors.append(err)
        weights = np.array([0.5, 0.3, 0.2])[:len(errors)]
        weights = weights / weights.sum()
        return np.sum([w * e for w, e in zip(weights, errors)])


def compute_hdrvdp2_metric(hdr_pred, hdr_gt):
    """
    Lightweight proxy for HDR-VDP-2 so training never breaks.
    Uses PSNR in log-encoded HDR space and maps it to ~[0,10].
    """
    hdr_pred_np = hdr_pred.detach().cpu().numpy()
    hdr_gt_np   = hdr_gt.detach().cpu().numpy()

    # Both are in [-1,1]. Map back to [0,10] cd/m^2-like range.
    hdr_pred_np = np.clip((hdr_pred_np + 1.0) * 5.0, 0, 10)
    hdr_gt_np   = np.clip((hdr_gt_np   + 1.0) * 5.0, 0, 10)

    # Log encoding to mimic HDR perception
    mu = 5000.0
    pred_tm = np.log(1 + mu * hdr_pred_np) / np.log(1 + mu)
    gt_tm   = np.log(1 + mu * hdr_gt_np)   / np.log(1 + mu)

    mse = np.mean((pred_tm - gt_tm) ** 2)
    psnr_val = 10.0 * np.log10(1.0 / (mse + 1e-12))

    # Normalize PSNR to a 0–10 “quality” score
    return float(np.clip(psnr_val / 10.0, 0.0, 10.0))


#def compute_tone_mapped_metrics(hdr_pred, hdr_gt, mu=5000.0):
#    hdr_pred_np = hdr_pred.cpu().numpy()
#    hdr_gt_np = hdr_gt.cpu().numpy()
#    
#    hdr_pred_np = (hdr_pred_np + 1.0) * 5.0
#    hdr_gt_np = (hdr_gt_np + 1.0) * 5.0
#    
#    hdr_pred_np = np.clip(hdr_pred_np, 0, 10)
#    hdr_gt_np = np.clip(hdr_gt_np, 0, 10)
#    
#    pred_tm = np.log(1 + mu * hdr_pred_np) / np.log(1 + mu)
#    gt_tm = np.log(1 + mu * hdr_gt_np) / np.log(1 + mu)
#    
#    pred_tm = (pred_tm * 255).astype(np.uint8)
#    gt_tm = (gt_tm * 255).astype(np.uint8)
#    
#    psnr_val = psnr(gt_tm, pred_tm, data_range=255)
#    
#    ssim_val = 0.0
#    for c in range(3):
#        ssim_val += ssim(gt_tm[c], pred_tm[c], data_range=255)
#    ssim_val /= 3
#    
#    return psnr_val, ssim_val


def save_sample_images(model, dataloader, epoch, device):
    model.eval()
    with torch.no_grad():
        ldr_gt, ldr_his, hdr_gt, filenames = next(iter(dataloader))
        ldr_gt = ldr_gt.to(device)
        ldr_his = ldr_his.to(device)
        hdr_gt = hdr_gt.to(device)
        
        hdr_pred = model(ldr_gt, ldr_his)
        
        ldr_save = (ldr_gt[0].cpu().permute(1, 2, 0).numpy() + 1) / 2.0
        hdr_pred_save = hdr_pred[0].cpu().permute(1, 2, 0).numpy()
        hdr_gt_save = hdr_gt[0].cpu().permute(1, 2, 0).numpy()

        # FHDR denormalize: inverse of (x - 0.5) / 0.5 is x * 0.5 + 0.5
        hdr_pred_save = hdr_pred_save * 0.5 + 0.5
        hdr_gt_save = hdr_gt_save * 0.5 + 0.5
        hdr_pred_save = np.clip(hdr_pred_save, 0, None)
        hdr_gt_save = np.clip(hdr_gt_save, 0, None)

        
        mu = 5000.0
        pred_tm = np.log(1 + mu * hdr_pred_save) / np.log(1 + mu)
        gt_tm = np.log(1 + mu * hdr_gt_save) / np.log(1 + mu)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ldr_save)
        axes[0].set_title('LDR Input')
        axes[0].axis('off')
        
        axes[1].imshow(pred_tm)
        axes[1].set_title(f'HDR Predicted (Epoch {epoch})')
        axes[1].axis('off')
        
        axes[2].imshow(gt_tm)
        axes[2].set_title('HDR Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(GENERATED_DIR, f'epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    model.train()


def validate(model, dataloader, criterion, device, hdrvdp_calculator):
    """
    Validation function with FHDR-style PSNR, SSIM, and HDR-VDP-2/3.
    """
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_hdrvdp2 = 0
    total_hdrvdp3 = 0
    num_images = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for ldr_gt, ldr_his, hdr_gt, _ in pbar:
            ldr_gt = ldr_gt.to(device)
            ldr_his = ldr_his.to(device)
            hdr_gt = hdr_gt.to(device)
            
            # Model prediction
            hdr_pred = model(ldr_gt, ldr_his)
            
            # Loss calculation
            loss, loss_dict = criterion(hdr_pred, hdr_gt)
            total_loss += loss.item()
            
            # FHDR-style PSNR: tonemap whole batch first (matches FHDR test.py)
            mu_tonemap_gt = mu_tonemap(hdr_gt)
            mu_tonemap_pred = mu_tonemap(hdr_pred)

            # Calculate metrics for each image in the batch
            for batch_ind in range(hdr_pred.size(0)):
                # PSNR: FHDR test.py - mse between mu_tonemap(pred), mu_tonemap(gt)
                mse = mse_loss(
                    mu_tonemap_pred[batch_ind:batch_ind + 1],
                    mu_tonemap_gt[batch_ind:batch_ind + 1]
                )
                psnr_val = 10 * np.log10(1.0 / mse.item())

                # SSIM: FHDR test.py - (output+1)/2, (gt+1)/2, compare_ssim(..., multichannel=True)
                generated = (
                    np.transpose(hdr_pred[batch_ind].cpu().numpy(), (1, 2, 0)) + 1
                ) / 2.0
                real = (
                    np.transpose(hdr_gt[batch_ind].cpu().numpy(), (1, 2, 0)) + 1
                ) / 2.0
                # win_size must be odd and <= min(H,W); default 7 can exceed small images
                h, w = generated.shape[0], generated.shape[1]
                min_side = min(h, w)
                if min_side < 1:
                    ssim_val = 0.0
                else:
                    win_size = min(7, min_side)
                    if win_size % 2 == 0:
                        win_size = max(1, win_size - 1)
                    ssim_val = compare_ssim(
                        generated, real,
                        channel_axis=-1 if generated.ndim == 3 else None,
                        win_size=win_size,
                        data_range=1.0
                    )




                
                # HDR-VDP-2 and HDR-VDP-3 (same as m_training / enhanced_model)
                pred_img = hdr_pred[batch_ind]
                gt_img = hdr_gt[batch_ind]
                hdrvdp2_val = hdrvdp_calculator.compute_hdrvdp2(pred_img, gt_img)
                hdrvdp3_val = hdrvdp_calculator.compute_hdrvdp3(pred_img, gt_img)
                
                total_psnr += psnr_val
                total_ssim += ssim_val
                total_hdrvdp2 += hdrvdp2_val
                total_hdrvdp3 += hdrvdp3_val
                num_images += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr_val:.2f}',
                    'ssim': f'{ssim_val:.4f}',
                    'hdrvdp2': f'{hdrvdp2_val:.4f}',
                    'hdrvdp3': f'{hdrvdp3_val:.4f}'
                })
    
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_hdrvdp2 = total_hdrvdp2 / num_images
    avg_hdrvdp3 = total_hdrvdp3 / num_images
    
    model.train()
    return avg_loss, avg_psnr, avg_ssim, avg_hdrvdp2, avg_hdrvdp3


def _find_latest_checkpoint(checkpoint_dir: str):
    """
    Return the path to the latest checkpoint (highest epoch), or None if none exist.
    Supports both `checkpoint_epoch_*.pth` and `best_model_epoch_*.pth`.
    """
    patterns = [
        os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"),
        os.path.join(checkpoint_dir, "best_model_epoch_*.pth"),
        os.path.join(checkpoint_dir, "latest.pth"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))

    if not candidates:
        return None

    def _score(path: str):
        base = os.path.basename(path)
        epoch = -1
        for token in base.replace(".pth", "").split("_"):
            if token.isdigit():
                epoch = int(token)
        mtime = os.path.getmtime(path)
        return (epoch, mtime)

    candidates.sort(key=_score)
    return candidates[-1]

def train(continue_training: bool = False, checkpoint_dir: str | None = None):
    print("=" * 80)
    print("HistoHDR-Net Training Script")
    print("=" * 80)
    print(f"LDR Directory: {LDR_DIR}")
    print(f"HDR Directory: {HDR_DIR}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("=" * 80)

    ckpt_dir = checkpoint_dir or CHECKPOINT_DIR

    # Ensure checkpoint and output dirs exist before training
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    print(f"Checkpoints will be saved to: {os.path.abspath(ckpt_dir)}")

    hdrvdp_calculator = HDRVDPMetrics(use_real_hdrvdp=False)
    
    train_dataset = HDRDataset(LDR_DIR, HDR_DIR, IMAGE_SIZE, mode='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    #init_hdrvdp2()
    
    val_split = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_split
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_split]
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    model = HistoHDRNet(pretrained=True).to(DEVICE)
    criterion = HistoHDRNetLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    start_epoch = 1
    best_psnr = 0.0
    best_epoch = 0

    if continue_training:
        latest_ckpt = _find_latest_checkpoint(ckpt_dir)
        if latest_ckpt is None:
            print(f"⚠ --continue-training was set, but no checkpoints found in: {os.path.abspath(ckpt_dir)}")
        else:
            print(f"Resuming from latest checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=DEVICE)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                except Exception as e:
                    print(f"⚠ Could not load scheduler state ({e}); continuing with fresh scheduler.")

            ckpt_epoch = int(checkpoint.get("epoch", 0))
            start_epoch = max(1, ckpt_epoch + 1)
            best_psnr = float(checkpoint.get("best_psnr", checkpoint.get("val_psnr", 0.0)) or 0.0)
            best_epoch = ckpt_epoch
            print(f"✓ Loaded checkpoint (epoch={ckpt_epoch}); continuing at epoch {start_epoch}")

    print("Running a dry-run validation before training...")
    try:
        val_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = validate(
            model, val_loader, criterion, DEVICE, hdrvdp_calculator
        )
        print(f"[Dry run] Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, "
            f"SSIM: {val_ssim:.4f}, HDR-VDP2: {val_hdrvdp2:.4f}, HDR-VDP3: {val_hdrvdp3:.4f}")
    except Exception as e:
        print("Dry-run validation failed with error:", e)
        raise
    
    csv_exists = os.path.exists(CSV_LOG_FILE)
    csv_file = open(CSV_LOG_FILE, 'a' if csv_exists else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    if not csv_exists:
        csv_writer.writerow([
            'epoch', 'train_loss', 'val_loss',
            'val_psnr', 'val_ssim', 'val_hdrvdp2', 'val_hdrvdp3',
            'l1_loss', 'vgg_loss', 'weber_loss', 'ms_ssim_loss', 'color_loss',
            'f1', 'recall', 'accuracy',
            'learning_rate', 'timestamp'
        ])

    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Training samples: {train_size}, Validation samples: {val_split}")
    print("=" * 80)
    
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        epoch_loss = 0
        epoch_components = {
            'l1': 0, 'vgg': 0, 'weber': 0, 'ms_ssim': 0, 'color': 0
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
        optimizer.zero_grad()
        
        for batch_idx, (ldr_gt, ldr_his, hdr_gt, _) in enumerate(pbar):
            ldr_gt = ldr_gt.to(DEVICE)
            ldr_his = ldr_his.to(DEVICE)
            hdr_gt = hdr_gt.to(DEVICE)
            
            hdr_pred = model(ldr_gt, ldr_his)
            loss, loss_dict = criterion(hdr_pred, hdr_gt)
            
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss_dict['total']
            for key in epoch_components:
                epoch_components[key] += loss_dict[key]
            
            pbar.set_postfix({
                'loss': f'{loss_dict["total"]:.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        scheduler.step()
        
        avg_train_loss = epoch_loss / len(train_loader)
        for key in epoch_components:
            epoch_components[key] /= len(train_loader)
        
        print(f"\nEpoch {epoch} Training Summary:")
        print(f"  Avg Loss: {avg_train_loss:.4f}")
        print(f"  L1: {epoch_components['l1']:.4f}, VGG: {epoch_components['vgg']:.4f}")
        print(f"  Weber: {epoch_components['weber']:.4f}, MS-SSIM: {epoch_components['ms_ssim']:.4f}")
        print(f"  Color: {epoch_components['color']:.4f}")
        
        val_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = validate(
            model, val_loader, criterion, DEVICE, hdrvdp_calculator
        )
        print(f" Validation - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, "
              f"HDR-VDP2: {val_hdrvdp2:.4f}, HDR-VDP3: {val_hdrvdp3:.4f}") 
        #print(f"  Validation - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        
        # F1, recall, accuracy: N/A for HDR regression (classification metrics)
        csv_writer.writerow([
            epoch, avg_train_loss, val_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3,
            epoch_components['l1'], epoch_components['vgg'],
            epoch_components['weber'], epoch_components['ms_ssim'],
            epoch_components['color'],
            '', '', '',  # f1, recall, accuracy
            optimizer.param_groups[0]['lr'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])
        csv_file.flush()

        # Save checkpoint at epoch 1 and every 5 epochs
        if epoch == 1 or epoch % 5 == 0:
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_loss': val_loss,
            }, ckpt_path)
            print(f"  ✓ Saved checkpoint at epoch {epoch} -> {ckpt_path}")
        
        if epoch % 10 == 0:
            save_sample_images(model, val_loader, epoch, DEVICE)
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                best_epoch = epoch
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_psnr': best_psnr,
                    'val_loss': val_loss,
                    'val_ssim': val_ssim
                }
                
                checkpoint_path = os.path.join(ckpt_dir, f'best_model_epoch_{epoch}.pth')
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✓ Saved best model (PSNR: {best_psnr:.2f}) at epoch {epoch}")
            else:
                print(f"  Current PSNR ({val_psnr:.2f}) < Best PSNR ({best_psnr:.2f} at epoch {best_epoch})")
        
        print("=" * 80)
        
        torch.cuda.empty_cache()
    
    csv_file.close()
    
    print("\nTraining completed!")
    print(f"Best PSNR: {best_psnr:.2f} at epoch {best_epoch}")
    print(f"Training log saved to: {CSV_LOG_FILE}")
    print(f"Checkpoints saved in: {ckpt_dir}")
    print(f"Generated images saved in: {GENERATED_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HistoHDR-Net training")
    parser.add_argument(
        "--continue-training",
        action="store_true",
        help="Resume from the latest checkpoint in ./checkpoints (by epoch).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory to use (overrides default).",
    )
    args = parser.parse_args()
    train(continue_training=args.continue_training, checkpoint_dir=args.checkpoint_dir)

