"""
Script for training the ArtHDRNet model with validation metrics and checkpointing.
"""

import os

# PyTorch 2.6+: ensure we can load full checkpoints (model+optimizer) with numpy objects
os.environ.pop("TORCH_FORCE_WEIGHTS_ONLY_LOAD", None)  # unset if present, so weights_only=False works
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm``
import argparse
import torch.nn.functional as F

# Import from FHDR code structure
from model import Dynamic_attention_model
from losses import EnhancedModelLoss
from options import Options
from util import (
    load_checkpoint,
    make_required_directories,
    mu_tonemap,
    save_checkpoint,
    save_hdr_image,
    save_ldr_image,
    update_lr,
)

from data_loader import HDRDataset
from image_transforms import LDRTransforms
from skimage.metrics import structural_similarity as compare_ssim

# For metrics calculation
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

class HDRVDPMetrics:
    """
    HDR-VDP metric calculator using FovVideoVDP (modern successor to HDR-VDP-2/3).
    Maintains exact same interface as before - works as drop-in replacement.
    
    Installation: pip install pyfvvdp
    """
    
    def __init__(self, use_real_hdrvdp=False):
        self.use_real_hdrvdp = use_real_hdrvdp
        self.fvvdp_model = None
        
        if use_real_hdrvdp:
            try:
                import pyfvvdp
                self.pyfvvdp = pyfvvdp
                
                # Initialize FovVideoVDP models for both metrics
                # HDR-VDP-2 equivalent: standard FHD display
                self.fvvdp2 = pyfvvdp.fvvdp(display_name='standard_fhd', heatmap=None)
                
                # HDR-VDP-3 equivalent: standard 4K display (higher quality)
                self.fvvdp3 = pyfvvdp.fvvdp(display_name='standard_4k', heatmap=None)
                
                self.hdrvdp_available = True
                print("✓ FovVideoVDP (HDR-VDP successor) loaded successfully")
                print("  Using standard_fhd for HDR-VDP-2 and standard_4k for HDR-VDP-3")
            except ImportError:
                print("⚠ WARNING: pyfvvdp not available. Using PU21-based proxy metrics.")
                print("  Install with: pip install pyfvvdp")
                print("  Falling back to PU21 perceptual encoding (from HDR-VDP-3)")
                self.hdrvdp_available = False
            except Exception as e:
                print(f"⚠ WARNING: Error initializing pyfvvdp: {e}")
                print("  Falling back to PU21 perceptual encoding")
                self.hdrvdp_available = False
        else:
            self.hdrvdp_available = False
            print("Using PU21-based perceptual metrics (fast proxy)")
    
    def compute_hdrvdp2(self, hdr_pred, hdr_gt):
        """
        Compute HDR-VDP-2 equivalent score.
        Uses FovVideoVDP with standard_fhd display if available, else PU21 encoding.
        
        Args:
            hdr_pred: Predicted HDR tensor (C, H, W) in [-1, 1]
            hdr_gt: Ground truth HDR tensor (C, H, W) in [-1, 1]
        
        Returns:
            float: Quality score (higher is better, typically 0-10 range)
        """
        if self.hdrvdp_available:
            return self._compute_real_fovvdp2(hdr_pred, hdr_gt)
        else:
            return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=5000.0)
    
    def compute_hdrvdp3(self, hdr_pred, hdr_gt):
        """
        Compute HDR-VDP-3 equivalent score.
        Uses FovVideoVDP with standard_4k display if available, else enhanced PU21.
        
        Args:
            hdr_pred: Predicted HDR tensor (C, H, W) in [-1, 1]
            hdr_gt: Ground truth HDR tensor (C, H, W) in [-1, 1]
        
        Returns:
            float: Quality score (higher is better, typically 0-10 range)
        """
        if self.hdrvdp_available:
            return self._compute_real_fovvdp3(hdr_pred, hdr_gt)
        else:
            return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=8000.0, use_spatial=True)
    
    def _compute_real_fovvdp2(self, hdr_pred, hdr_gt):
        """Real FovVideoVDP with FHD display (HDR-VDP-2 equivalent)"""
        try:
            # Convert from PyTorch tensor to numpy
            pred_np = hdr_pred.detach().cpu().numpy()
            gt_np = hdr_gt.detach().cpu().numpy()
            
            # Convert from [-1, 1] to absolute luminance [0, 100] cd/m²
            pred_np = np.clip((pred_np + 1.0) * 50.0, 0.01, 100.0)
            gt_np = np.clip((gt_np + 1.0) * 50.0, 0.01, 100.0)
            
            # Transpose to HWC format (pyfvvdp expects HWC)
            pred_np = np.transpose(pred_np, (1, 2, 0))
            gt_np = np.transpose(gt_np, (1, 2, 0))
            
            # Compute FovVideoVDP
            Q_JOD, stats = self.fvvdp2.predict(pred_np, gt_np, dim_order='HWC')
            
            # Convert JOD to 0-10 scale (JOD typically ranges from -inf to 10)
            # Higher JOD = better quality
            q_score = float(np.clip(Q_JOD, 0.0, 10.0))
            
            return q_score
            
        except Exception as e:
            print(f"Warning: FovVideoVDP-2 failed ({e}), using fallback")
            return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=5000.0)
    
    def _compute_real_fovvdp3(self, hdr_pred, hdr_gt):
        """Real FovVideoVDP with 4K display (HDR-VDP-3 equivalent)"""
        try:
            # Convert from PyTorch tensor to numpy
            pred_np = hdr_pred.detach().cpu().numpy()
            gt_np = hdr_gt.detach().cpu().numpy()
            
            # Convert from [-1, 1] to absolute luminance [0, 100] cd/m²
            pred_np = np.clip((pred_np + 1.0) * 50.0, 0.01, 100.0)
            gt_np = np.clip((gt_np + 1.0) * 50.0, 0.01, 100.0)
            
            # Transpose to HWC format
            pred_np = np.transpose(pred_np, (1, 2, 0))
            gt_np = np.transpose(gt_np, (1, 2, 0))
            
            # Compute FovVideoVDP
            Q_JOD, stats = self.fvvdp3.predict(pred_np, gt_np, dim_order='HWC')
            
            # Convert JOD to 0-10 scale
            q_score = float(np.clip(Q_JOD, 0.0, 10.0))
            
            return q_score
            
        except Exception as e:
            print(f"Warning: FovVideoVDP-3 failed ({e}), using fallback")
            return self._compute_pu21_metric(hdr_pred, hdr_gt, mu=8000.0, use_spatial=True)
    
    def _compute_pu21_metric(self, hdr_pred, hdr_gt, mu=5000.0, use_spatial=False):
        """
        PU21 perceptual encoding from HDR-VDP-3 (high-quality fallback).
        PU21 = Perceptually Uniform 2021 encoding.
        
        Args:
            mu: Tone mapping parameter (5000 for VDP-2 style, 8000 for VDP-3 style)
            use_spatial: Whether to use spatial pooling (VDP-3 style)
        """
        pred_np = hdr_pred.detach().cpu().numpy()
        gt_np = hdr_gt.detach().cpu().numpy()
        
        # Convert from [-1, 1] to luminance [0, 100] cd/m²
        L_pred = np.clip((pred_np + 1.0) * 50.0, 0.01, 100.0)
        L_gt = np.clip((gt_np + 1.0) * 50.0, 0.01, 100.0)
        
        # PU21 encoding (perceptually uniform)
        # This is the actual encoding used in HDR-VDP-3
        pu_pred = self._pu21_encode(L_pred)
        pu_gt = self._pu21_encode(L_gt)
        
        # Compute MSE in perceptually uniform space
        mse = np.mean((pu_pred - pu_gt) ** 2)
        
        # Add spatial pooling for VDP-3 style
        if use_spatial:
            # Multi-scale spatial pooling (mimics CSF filtering)
            spatial_mse = self._compute_spatial_error(pu_pred, pu_gt)
            mse = 0.6 * mse + 0.4 * spatial_mse
        
        # Convert to PSNR-like quality score
        max_val = self._pu21_encode(np.array([100.0]))[0]  # Max possible value
        psnr = 10.0 * np.log10((max_val ** 2) / (mse + 1e-10))
        
        # Normalize to 0-10 range
        q_score = float(np.clip(psnr / 10.0, 0.0, 10.0))
        
        return q_score
    
    def _pu21_encode(self, L):
        """
        PU21 encoding from HDR-VDP-3.
        Maps luminance to perceptually uniform space.
        
        L: luminance in cd/m² (absolute units)
        Returns: perceptually encoded values
        """
        # PU21 encoding formula from Mantiuk et al. 2023
        # This is the actual formula used in HDR-VDP-3
        P = np.log((L + 1e-4) / (L + 0.01))
        return P
    
    def _compute_spatial_error(self, pred, gt, scales=[1, 2, 4]):
        """
        Multi-scale spatial error (mimics CSF in HDR-VDP-3).
        Computes error at different spatial scales.
        """
        errors = []
        
        for scale in scales:
            if scale == 1:
                err = np.mean((pred - gt) ** 2)
            else:
                # Downsample
                pred_down = pred[:, ::scale, ::scale]
                gt_down = gt[:, ::scale, ::scale]
                err = np.mean((pred_down - gt_down) ** 2)
            
            errors.append(err)
        
        # Weighted average (lower frequencies weighted more)
        weights = np.array([0.5, 0.3, 0.2])[:len(errors)]
        weights = weights / weights.sum()
        
        spatial_mse = np.sum([w * e for w, e in zip(weights, errors)])
        
        return spatial_mse

def compute_psnr_ssim(pred, gt):
    """
    Compute PSNR-μ and SSIM matching FHDR implementation.
    
    Args:
        pred: Predicted HDR (C, H, W) in [-1, 1]
        gt: Ground truth HDR (C, H, W) in [-1, 1]
    
    Returns:
        psnr: PSNR-μ in dB
        ssim: SSIM value
    """

    pred_batch = pred.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    
    # PSNR calculation
    mu_tonemap_gt = mu_tonemap(gt_batch)
    mu_tonemap_pred = mu_tonemap(pred_batch)
    mse = F.mse_loss(mu_tonemap_pred, mu_tonemap_gt)
    psnr = 10 * np.log10(1 / mse.item())
    
    # SSIM calculation - work with single image (C, H, W)
    generated = (np.transpose(pred.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    real = (np.transpose(gt.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    ssim = compare_ssim(generated, real, channel_axis=-1, win_size=7, data_range=1.0)

    return psnr, ssim


def save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3):
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_hdrvdp2', 'val_hdrvdp3']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'val_psnr': f"{val_psnr:.4f}",
            'val_ssim': f"{val_ssim:.4f}",
            'val_hdrvdp2': f"{val_hdrvdp2:.4f}",
            'val_hdrvdp3': f"{val_hdrvdp3:.4f}"
        })

def unwrap_loss(loss_out):
    if isinstance(loss_out, (tuple, list)):
        return loss_out[0]
    if isinstance(loss_out, dict):
        return loss_out["loss"]
    return loss_out


def sanity_check(model, criterion, optimizer, train_loader, val_loader, device, hdrvdp_calculator):

    # Initialize transformer
    ldr_transformer = LDRTransforms()
    
    # ---- one TRAIN step ----
    model.train()
    data = next(iter(train_loader))

    input_ldr = data["ldr_image"].to(device)
    ground_truth = data["hdr_image"].to(device)

    # Apply transformations
    original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)
    
    # Forward pass with 6 inputs
    outputs = model(underexposed, overexposed, original)
    loss_out = criterion(outputs, ground_truth)

    loss = unwrap_loss(loss_out)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    with torch.no_grad():
        pred = outputs[-1]
        print("[SANITY][TRAIN] loss:", float(loss.item()))
        print("[SANITY][TRAIN] shapes:",
              "ldr", tuple(input_ldr.shape),
              "gt", tuple(ground_truth.shape),
              "pred", tuple(pred.shape))
        # quick “dummy HDR” detection (all-ones -> near-zero std after normalization)
        print("[SANITY][TRAIN] gt std:", float(ground_truth.std().item()))

    # ---- one VAL step (validate_model logic, but just 1 batch) ----
    model.eval()
    with torch.no_grad():
        data = next(iter(val_loader))
        input_ldr = data["ldr_image"].to(device)
        ground_truth = data["hdr_image"].to(device)
        
        # Apply transformations
        original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)
        
        # Forward pass with 6 inputs
        outputs = model(underexposed, overexposed, original)
        pred = outputs

        # IMPORTANT: use (C,H,W) not (1,C,H,W)
        pred_img = pred[0]
        gt_img = ground_truth[0]

        psnr, ssim = compute_psnr_ssim(pred_img, gt_img)

        hdrvdp2 = hdrvdp_calculator.compute_hdrvdp2(pred_img, gt_img)
        hdrvdp3 = hdrvdp_calculator.compute_hdrvdp3(pred_img, gt_img)
        print("[SANITY][VAL] psnr:", float(psnr), "ssim:", float(ssim))
        print("[SANITY][VAL] hdrvdp2:", float(hdrvdp2), "hdrvdp3:", float(hdrvdp3))

    model.train()

def validate_model(model, val_loader, device, epoch, hdrvdp_calculator, save_samples=False):
    """
    Validate model on validation set and compute metrics.
    """
    model.eval()
    
    # Initialize transformer
    ldr_transformer = LDRTransforms()
    
    total_psnr = 0.0
    total_ssim = 0.0 
    total_hdrvdp2 = 0.0
    total_hdrvdp3 = 0.0
    num_samples = 0

    # Directory for saving validation samples
    if save_samples:
        sample_dir = f"./validation_results/epoch_{epoch}"
        os.makedirs(sample_dir, exist_ok=True)
        sample_count = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc="Validation")):

            input_ldr = data["ldr_image"].to(device)
            ground_truth = data["hdr_image"].to(device)
            
            # Apply transformations
            original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)

            
            # Forward pass with 6 inputs
            outputs = model(underexposed, overexposed, original)
            hdr_pred = outputs
            
            # Save samples (first 10 images)
            if save_samples and sample_count < 10:
                for i in range(min(hdr_pred.shape[0], 10 - sample_count)):
                    # Save LDR image
                    ldr_path = os.path.join(sample_dir, f"ldr_{sample_count}.png")
                    save_ldr_image(
                        img_tensor=input_ldr,
                        batch=i,
                        path=ldr_path
                    )
                    
                    # Save predicted HDR
                    pred_path = os.path.join(sample_dir, f"pred_hdr_{sample_count}.hdr")
                    save_hdr_image(
                        img_tensor=hdr_pred,
                        batch=i,
                        path=pred_path
                    )
                    
                    # Save ground truth HDR
                    gt_path = os.path.join(sample_dir, f"gt_hdr_{sample_count}.hdr")
                    save_hdr_image(
                        img_tensor=ground_truth,
                        batch=i,
                        path=gt_path
                    )
                    
                    sample_count += 1
            
            # Calculate metrics for each image in batch
            for i in range(hdr_pred.shape[0]):
                pred_img = hdr_pred[i]
                gt_img = ground_truth[i]
                
                # Compute PSNR and SSIM
                psnr, ssim = compute_psnr_ssim(pred_img, gt_img)
                total_psnr += psnr
                total_ssim += ssim
                
                # Compute HDR-VDP-2 proxy
                hdrvdp2 = hdrvdp_calculator.compute_hdrvdp2(pred_img, gt_img)
                hdrvdp3 = hdrvdp_calculator.compute_hdrvdp3(pred_img, gt_img)
                total_hdrvdp2 += hdrvdp2
                total_hdrvdp3 += hdrvdp3
                
                num_samples += 1
    
    # Calculate averages
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0

    avg_hdrvdp2 = total_hdrvdp2 / num_samples if num_samples > 0 else 0
    avg_hdrvdp3 = total_hdrvdp3 / num_samples if num_samples > 0 else 0
    model.train()
    return avg_psnr, avg_ssim, avg_hdrvdp2, avg_hdrvdp3

def main():
    # Initialize options
    opt = Options().parse()
    
    # Dataset paths
    LDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"
    HDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"
    CHECKPOINT_DIR = "checkpoints"
    CSV_FILE = "training_metrics.csv"
    LATEST_CKPT = os.path.join(CHECKPOINT_DIR, 'latest.pth')   # <-- ADD THIS LINE
    
    # Create directories
    make_required_directories(mode="train")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("./validation_results", exist_ok=True)
    
    # Initialize HDR-VDP calculator
    hdrvdp_calculator = HDRVDPMetrics(use_real_hdrvdp=False)
    
    # ======================================
    # Load and split dataset
    # ======================================
    print("Loading dataset...")
    full_dataset = HDRDataset(LDR_DIR, HDR_DIR, mode="train")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # ========================================
    # Set device based on gpu_ids
    # ========================================
    if opt.gpu_ids == "-1":
        device = torch.device('cpu')
        print("Using CPU")
    else:
        # Use first GPU (single GPU)
        device = torch.device('cuda:0')
        print(f"Using GPU: {opt.gpu_ids}")
    
    # Initialize model
    model = Dynamic_attention_model(256, 512, 1024, 2048).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr)
    
    # Loss function
    criterion = EnhancedModelLoss().to(device)
    
    # ========================================
    # SANITY CHECK: Run validation before training
    # ========================================
    print("\n" + "="*60)
    print("Running pre-training validation sanity check...")
    print("="*60)
    
    try:
        # Test validation on first 10 samples
        val_subset = torch.utils.data.Subset(val_dataset, range(min(10, len(val_dataset))))
        val_subset_loader = DataLoader(val_subset, batch_size=1, shuffle=False)
        
        print("Testing validation pipeline on 10 samples...")
        test_psnr, test_ssim, test_hdrvdp2, test_hdrvdp3 = validate_model(
            model, val_subset_loader, device, 0, hdrvdp_calculator, save_samples=False
        )
        
        print(f"✓ Validation sanity check PASSED!")
        print(f"  Sample PSNR: {test_psnr:.4f}")
        print(f"  Sample SSIM: {test_ssim:.4f}")
        print(f"  Sample HDR-VDP-2: {test_hdrvdp2:.4f}")
        print(f"  Sample HDR-VDP-3: {test_hdrvdp3:.4f}")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ VALIDATION SANITY CHECK FAILED!")
        print(f"Error: {e}")
        print("\nFix the error above before starting training.")
        import traceback
        traceback.print_exc()
        return  # Exit before training
    
    # ========================================
    # Load checkpoint if continuing training
    # ========================================
    start_epoch = 1
    best_val_psnr = 0
    best_val_ssim = 0
    best_val_hdrvdp2 = 0
    best_val_hdrvdp3 = 0
    
    if opt.continue_train:
        latest_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'latest.pth')
        if os.path.isfile(latest_checkpoint_path):
            try:
                # PyTorch 2.6+: allowlist numpy types in optimizer state (overrides weights_only when env forces it)
                _safe_globals = []
                try:
                    import numpy.core.multiarray as _np_multiarray
                    _safe_globals.extend([_np_multiarray.scalar, getattr(_np_multiarray, "_reconstruct", None)])
                except (ImportError, AttributeError):
                    pass
                _safe_globals = [g for g in _safe_globals if g is not None]
                if _safe_globals:
                    torch.serialization.add_safe_globals(_safe_globals)
                checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_psnr = checkpoint.get('best_val_psnr', 0)
                best_val_ssim = checkpoint.get('best_val_ssim', 0)
                best_val_hdrvdp2 = checkpoint.get('best_val_hdrvdp2', 0)
                best_val_hdrvdp3 = checkpoint.get('best_val_hdrvdp3', 0)
                print(f"Resuming training from epoch {start_epoch} (latest checkpoint)")
            except Exception as e:
                print(f"Error loading latest checkpoint: {e}. Training from scratch.")
                start_epoch = 1
        else:
            print("No latest checkpoint found. Training from scratch.")

    # ========================================
    # Training loop
    # ========================================
    print("\nStarting training on device:", device)
    
    for epoch in range(start_epoch, opt.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0
        
        # Check whether LR needs to be updated
        if epoch > opt.lr_decay_after:
            for param_group in optimizer.param_groups:
                lr_scale = 1.0 - max(0, epoch - opt.lr_decay_after) / (opt.epochs - opt.lr_decay_after)
                param_group['lr'] = opt.lr * lr_scale
        
        print(f"\nEpoch: {epoch}/{opt.epochs}")
        
        # Training phase
        model.train()
        ldr_transformer = LDRTransforms(
            gamma_value=2.2,
            underexposed_ev=-2.0,
            overexposed_ev=2.0,
            clahe_clip_limit=2.0,
            clahe_tile_size=8
        )
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training")):
            input_ldr = data["ldr_image"].to(device)
            ground_truth = data["hdr_image"].to(device)
            
            # Apply transformations
            original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)

            # Forward pass
            outputs = model(underexposed, overexposed, original)
            loss_out = criterion(outputs, ground_truth)
            loss = unwrap_loss(loss_out)
            
            running_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            num_batches += 1
            
            # Log batch information
            if (batch_idx + 1) % opt.log_after == 0:
                avg_loss = running_loss / num_batches
                tqdm.write(f"  Batch: {batch_idx + 1}; Training loss: {avg_loss:.6f}")
        
        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start
        
        # Validation phase
        print("Validating...")
        val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = validate_model(
            model, val_loader, device, epoch, hdrvdp_calculator,
            save_samples=(epoch % opt.save_ckpt_after == 0 or epoch == 1)
        )

        # Save metrics to CSV
        save_metrics_to_csv(CSV_FILE, epoch, avg_train_loss, val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3)
        
        # Print epoch results
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{opt.epochs} Summary")
        print(f"{'='*60}")
        print(f"  Training Loss    : {avg_train_loss:.6f}")
        print(f"  Validation PSNR  : {val_psnr:.4f} dB")
        print(f"  Validation SSIM  : {val_ssim:.4f}")
        print(f"  HDR-VDP-2 Score  : {val_hdrvdp2:.4f}")
        print(f"  HDR-VDP-3 Score  : {val_hdrvdp3:.4f}")
        print(f"  Epoch Time       : {epoch_time:.2f} seconds")
        print(f"{'='*60}")
        
        # Save checkpoint
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_val_ssim = val_ssim
            best_val_hdrvdp2 = val_hdrvdp2
            best_val_hdrvdp3 = val_hdrvdp3
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_psnr': best_val_psnr,
                'best_val_ssim': best_val_ssim,
                'best_val_hdrvdp2': best_val_hdrvdp2,   # <-- added
                'best_val_hdrvdp3': best_val_hdrvdp3,   # <-- added
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_hdrvdp2': val_hdrvdp2,             # <-- added
                'val_hdrvdp3': val_hdrvdp3              # <-- added
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  ✓ Saved best model with PSNR: {val_psnr:.4f}")

        if epoch % opt.save_ckpt_after == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, f'epoch_{epoch}.pth'))
            print(f"  ✓ Saved checkpoint at epoch {epoch}")

                # Save latest checkpoint (always overwrite)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_psnr': best_val_psnr,
            'best_val_ssim': best_val_ssim,
            'best_val_hdrvdp2': best_val_hdrvdp2,
            'best_val_hdrvdp3': best_val_hdrvdp3,
        }, LATEST_CKPT)
        print(f"  ✓ Saved latest checkpoint")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"  Best PSNR     : {best_val_psnr:.4f} dB")
    print(f"  Best SSIM     : {best_val_ssim:.4f}")
    print(f"  Best HDR-VDP-2: {best_val_hdrvdp2:.4f}")
    print(f"  Best HDR-VDP-3: {best_val_hdrvdp3:.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
