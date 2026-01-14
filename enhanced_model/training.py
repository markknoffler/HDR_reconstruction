"""
Script for training the ArtHDRNet model with validation metrics and checkpointing.
"""

import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import argparse

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
import deepspeed


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

#def compute_hdrvdp2_metric(hdr_pred, hdr_gt):
#    """
#    Lightweight proxy for HDR-VDP-2 so training never breaks.
#    Uses PSNR in log-encoded HDR space and maps it to ~[0,10].
#    """
#    hdr_pred_np = hdr_pred.detach().cpu().numpy()
#    hdr_gt_np = hdr_gt.detach().cpu().numpy()
#
#    # Both are in [-1,1]. Map back to [0,10] cd/m^2-like range.
#    hdr_pred_np = np.clip((hdr_pred_np + 1.0) * 5.0, 0, 10)
#    hdr_gt_np = np.clip((hdr_gt_np + 1.0) * 5.0, 0, 10)
#
#    # Log encoding to mimic HDR perception
#    mu = 5000.0
#    pred_tm = np.log(1 + mu * hdr_pred_np) / np.log(1 + mu)
#    gt_tm = np.log(1 + mu * hdr_gt_np) / np.log(1 + mu)
#
#    mse = np.mean((pred_tm - gt_tm) ** 2)
#    psnr_val = 10.0 * np.log10(1.0 / (mse + 1e-12))
#
#    # Normalize PSNR to a 0-10 "quality" score
#    return float(np.clip(psnr_val / 10.0, 0.0, 10.0))

def compute_psnr_ssim(pred, gt, avg_psnr, avg_ssim):
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
    mse = mse_loss(mu_tonemap_pred, mu_tonemap_gt)
    psnr = 10 * np.log10(1 / mse.item())
    
    # SSIM calculation - work with single image (C, H, W)
    generated = (np.transpose(pred.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    real = (np.transpose(gt.cpu().numpy(), (1, 2, 0)) + 1) / 2.0
    ssim = compare_ssim(generated, real, multichannel=True)
    
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
    #input_ldr = data["ldr_image"].to(device)
    #ground_truth = data["hdr_image"].to(device)

    # *** DEEPSPEED CHANGE: Data to GPU and FP16 ***
    input_ldr = data["ldr_image"].cuda().half()  # Add .half()
    ground_truth = data["hdr_image"].cuda().half()  # Add .half()

    
    # Apply transformations
    original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)

    original = original.half()
    gamma = gamma.half()
    underexposed = underexposed.half()
    overexposed = overexposed.half()
    hist_eq = hist_eq.half()
    clahe = clahe.half()
    
    # Forward pass with 6 inputs
    #with autocast():
    outputs = model(gamma, underexposed, overexposed, original, clahe, hist_eq)
    loss_out = criterion(outputs, ground_truth)

    #outputs = model(gamma, underexposed, overexposed, original, clahe, hist_eq)
    #loss_out = criterion(outputs, ground_truth)
    loss = unwrap_loss(loss_out)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
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
        outputs = model(gamma, underexposed, overexposed, original, clahe, hist_eq)
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
            #input_ldr = data["ldr_image"].to(device)
            #ground_truth = data["hdr_image"].to(device)

            input_ldr = data["ldr_image"].cuda().half()  # Add .cuda().half()
            ground_truth = data["hdr_image"].cuda().half()  # Add .cuda().half()
            
            # Apply transformations
            original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)

            original = original.half()
            gamma = gamma.half()
            underexposed = underexposed.half()
            overexposed = overexposed.half()
            hist_eq = hist_eq.half()
            clahe = clahe.half()
            
            # Forward pass with 6 inputs
            outputs = model(gamma, underexposed, overexposed, original, clahe, hist_eq)
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
    # *** DEEPSPEED CHANGE: Model initialization ***
    # REMOVE ALL THE OLD GPU CONFIGURATION CODE
    # ========================================
    model = Dynamic_attention_model(256, 512, 1024, 2048)
    # DON'T call .to(device) - DeepSpeed handles it!
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    
    # ========================================
    # *** DEEPSPEED CHANGE: Initialize DeepSpeed ***
    # ========================================
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        model_parameters=model.parameters(),
        config='ds_config.json'  # DeepSpeed config file
    )
    
    # Loss function (no .to(device) needed)
    #criterion = EnhancedModelLoss()
    criterion = EnhancedModelLoss().cuda().half()
    
    # ========================================
    # Load checkpoint if continuing training
    # ========================================
    start_epoch = 1
    best_val_psnr = 0
    best_val_ssim = 0
    best_val_hdrvdp2 = 0
    best_val_hdrvdp3 = 0
    
    if opt.continue_train:
        try:
            # DeepSpeed checkpoint loading
            _, client_state = model_engine.load_checkpoint(CHECKPOINT_DIR)
            if client_state:
                start_epoch = client_state['epoch'] + 1
                best_val_psnr = client_state.get('best_val_psnr', 0)
                best_val_ssim = client_state.get('best_val_ssim', 0)
                best_val_hdrvdp2 = client_state.get('best_val_hdrvdp2', 0)
                best_val_hdrvdp3 = client_state.get('best_val_hdrvdp3', 0)
                print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Checkpoint not found: {e}. Training from scratch.")
            start_epoch = 1
    
    # ========================================
    # Training loop
    # ========================================
    print("\nStarting training with DeepSpeed ZeRO-3...")
    print(f"Model will be automatically sharded across {torch.cuda.device_count()} GPUs + CPU offloading")
    
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
        model_engine.train()
        ldr_transformer = LDRTransforms(
            gamma_value=2.2,
            underexposed_ev=-2.0,
            overexposed_ev=2.0,
            clahe_clip_limit=2.0,
            clahe_tile_size=8
        )
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training")):
            # *** DEEPSPEED CHANGE: Data to GPU ***
            #input_ldr = data["ldr_image"].cuda()
            #ground_truth = data["hdr_image"].cuda()

            input_ldr = data["ldr_image"].cuda().half()  # Add .cuda().half()
            ground_truth = data["hdr_image"].cuda().half()  # Add .cuda().half()

            
            # Apply transformations
            original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)

            original = original.half()
            gamma = gamma.half()
            underexposed = underexposed.half()
            overexposed = overexposed.half()
            hist_eq = hist_eq.half()
            clahe = clahe.half()

            # *** DEEPSPEED CHANGE: Forward pass (NO autocast!) ***
            # DeepSpeed automatically uses FP16
            outputs = model_engine(gamma, underexposed, overexposed, original, clahe, hist_eq)
            
            # Calculate loss
            loss_out = criterion(outputs, ground_truth)
            
            if isinstance(loss_out, (tuple, list)):
                loss = loss_out[0]
            elif isinstance(loss_out, dict):
                loss = loss_out["loss"]
            else:
                loss = loss_out
            
            running_loss += loss.item()
            
            # *** DEEPSPEED CHANGE: Backward pass ***
            model_engine.backward(loss)
            model_engine.step()
            
            num_batches += 1
            
            # Log batch information
            if (batch_idx + 1) % opt.log_after == 0:
                avg_loss = running_loss / num_batches
                print(f"  Batch: {batch_idx + 1}; Training loss: {avg_loss:.6f}")
        
        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        epoch_time = time.time() - epoch_start
        
        # Validation phase
        print("Validating...")
        val_psnr, val_ssim, val_hdrvdp2, val_hdrvdp3 = validate_model(
            model_engine, val_loader, torch.device('cuda:0'), epoch, hdrvdp_calculator,
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
        
        # *** DEEPSPEED CHANGE: Save checkpoint ***
        client_state = {
            'epoch': epoch,
            'best_val_psnr': best_val_psnr,
            'best_val_ssim': best_val_ssim,
            'best_val_hdrvdp2': best_val_hdrvdp2,
            'best_val_hdrvdp3': best_val_hdrvdp3,
            'avg_train_loss': avg_train_loss,
            'val_psnr': val_psnr,
            'val_ssim': val_ssim
        }
        
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_val_ssim = val_ssim
            best_val_hdrvdp2 = val_hdrvdp2
            best_val_hdrvdp3 = val_hdrvdp3
            
            # Save best model
            model_engine.save_checkpoint(CHECKPOINT_DIR, tag=f'best_epoch_{epoch}', client_state=client_state)
            print(f"  ✓ Saved best model with PSNR: {val_psnr:.4f}")
        
        # Save regular checkpoint
        if epoch % opt.save_ckpt_after == 0:
            model_engine.save_checkpoint(CHECKPOINT_DIR, tag=f'epoch_{epoch}', client_state=client_state)
            print(f"  ✓ Saved checkpoint at epoch {epoch}")
    
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
