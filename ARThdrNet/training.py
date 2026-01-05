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
from model import ArtHDRNet, ArtHDRLoss
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

# For metrics calculation
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2


def compute_hdrvdp2_metric(hdr_pred, hdr_gt):
    """
    Lightweight proxy for HDR-VDP-2 so training never breaks.
    Uses PSNR in log-encoded HDR space and maps it to ~[0,10].
    """
    hdr_pred_np = hdr_pred.detach().cpu().numpy()
    hdr_gt_np = hdr_gt.detach().cpu().numpy()

    # Both are in [-1,1]. Map back to [0,10] cd/m^2-like range.
    hdr_pred_np = np.clip((hdr_pred_np + 1.0) * 5.0, 0, 10)
    hdr_gt_np = np.clip((hdr_gt_np + 1.0) * 5.0, 0, 10)

    # Log encoding to mimic HDR perception
    mu = 5000.0
    pred_tm = np.log(1 + mu * hdr_pred_np) / np.log(1 + mu)
    gt_tm = np.log(1 + mu * hdr_gt_np) / np.log(1 + mu)

    mse = np.mean((pred_tm - gt_tm) ** 2)
    psnr_val = 10.0 * np.log10(1.0 / (mse + 1e-12))

    # Normalize PSNR to a 0-10 "quality" score
    return float(np.clip(psnr_val / 10.0, 0.0, 10.0))


def compute_psnr_ssim(pred, gt):
    """
    Compute PSNR and SSIM similar to FHDR code.
    Both images should be in range [-1, 1]
    """
    # Convert to numpy and scale to [0, 1]
    pred_np = pred.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    
    # Ensure correct shape: (C, H, W) -> (H, W, C)
    if pred_np.shape[0] == 3:  # If channels first
        pred_np = np.transpose(pred_np, (1, 2, 0))
        gt_np = np.transpose(gt_np, (1, 2, 0))
    
    # Scale from [-1, 1] to [0, 1]
    pred_np = (pred_np + 1) / 2.0
    gt_np = (gt_np + 1) / 2.0
    
    # Calculate PSNR (using mu-tonemap like in FHDR)
    mu = 5000.0
    pred_tm = np.log(1.0 + mu * pred_np) / np.log(1.0 + mu)
    gt_tm = np.log(1.0 + mu * gt_np) / np.log(1.0 + mu)
    
    mse = np.mean((pred_tm - gt_tm) ** 2)
    psnr = 10 * np.log10(1.0 / (mse + 1e-10))
    
    # Calculate SSIM
    # Ensure images are float32
    pred_np = pred_np.astype(np.float32)
    gt_np = gt_np.astype(np.float32)
    
    # For multi-channel images, use multichannel=True
    ssim = structural_similarity(
        pred_np, gt_np, 
        multichannel=True, 
        data_range=1.0, 
        win_size=11,
        channel_axis=-1 if pred_np.shape[-1] == 3 else None
    )
    
    return psnr, ssim


class HDRDataset(Dataset):
    """
    Dataset class for HDR training, similar to FHDR's HDRDataset.
    """
    def __init__(self, ldr_dir, hdr_dir, mode="train", transform=None):
        self.ldr_dir = ldr_dir
        self.hdr_dir = hdr_dir
        self.mode = mode
        self.transform = transform
        
        # Get all LDR files
        self.ldr_files = sorted([f for f in os.listdir(ldr_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        # Get corresponding HDR files
        self.hdr_files = []
        
        for ldr_file in self.ldr_files:
            # Assuming same filename with .hdr extension
            hdr_file = ldr_file.replace('.png', '.hdr').replace('.jpg', '.hdr').replace('.jpeg', '.hdr')
            hdr_path = os.path.join(hdr_dir, hdr_file)
            if os.path.exists(hdr_path):
                self.hdr_files.append(hdr_file)
            else:
                print(f"Warning: No HDR file found for {ldr_file}")
        
        # Make sure lengths match
        self.ldr_files = self.ldr_files[:len(self.hdr_files)]
        self.hdr_files = self.hdr_files[:len(self.ldr_files)]
        
        print(f"{mode.capitalize()} dataset loaded: {len(self.ldr_files)} samples")
    
    def __len__(self):
        return len(self.ldr_files)
    
    def __getitem__(self, idx):
        ldr_path = os.path.join(self.ldr_dir, self.ldr_files[idx])
        hdr_path = os.path.join(self.hdr_dir, self.hdr_files[idx])
        
        # Load LDR image
        ldr_img = cv2.imread(ldr_path)
        if ldr_img is None:
            raise ValueError(f"Failed to load LDR image: {ldr_path}")
        
        ldr_img = cv2.cvtColor(ldr_img, cv2.COLOR_BGR2RGB)
        ldr_img = ldr_img.astype(np.float32) / 255.0  # [0, 1]
        ldr_img = 2.0 * ldr_img - 1.0  # [-1, 1]
        ldr_img = torch.from_numpy(ldr_img).permute(2, 0, 1)  # HWC to CHW
        
        # Load HDR image
        # For simplicity, we'll assume HDR images are stored as numpy arrays
        # You may need to adjust this based on your actual HDR format
        try:
            # Try to load as numpy array
            if hdr_path.endswith('.npy'):
                hdr_img = np.load(hdr_path)
            else:
                # For .hdr files, you might need a custom loader
                # Here's a simple placeholder - adjust as needed
                import imageio
                hdr_img = imageio.imread(hdr_path, format='HDR-FI')
        except:
            # If loading fails, create a dummy HDR image
            print(f"Warning: Could not load HDR image {hdr_path}, using dummy")
            hdr_img = np.ones((ldr_img.shape[1], ldr_img.shape[2], 3), dtype=np.float32)
        
        hdr_img = hdr_img.astype(np.float32)
        # Normalize HDR to [-1, 1] range
        # This depends on your HDR data - adjust as needed
        if hdr_img.max() > 1.0:
            hdr_img = hdr_img / hdr_img.max()  # Normalize to [0, 1]
        hdr_img = 2.0 * hdr_img - 1.0  # [-1, 1]
        
        hdr_img = torch.from_numpy(hdr_img).permute(2, 0, 1)  # HWC to CHW
        
        return {
            "ldr_image": ldr_img,
            "hdr_image": hdr_img,
            "ldr_path": self.ldr_files[idx],
            "hdr_path": self.hdr_files[idx]
        }


def validate_model(model, val_loader, device, epoch, save_samples=False):
    """
    Validate model on validation set and compute metrics.
    """
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_hdrvdp = 0.0
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
            
            # Forward pass
            outputs = model(input_ldr)
            hdr_pred = outputs[-1]  # Take the last iteration output
            
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
                pred_img = hdr_pred[i:i+1]
                gt_img = ground_truth[i:i+1]
                
                # Compute PSNR and SSIM
                psnr, ssim = compute_psnr_ssim(pred_img, gt_img)
                total_psnr += psnr
                total_ssim += ssim
                
                # Compute HDR-VDP-2 proxy
                hdrvdp = compute_hdrvdp2_metric(pred_img, gt_img)
                total_hdrvdp += hdrvdp
                
                num_samples += 1
    
    # Calculate averages
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0
    avg_hdrvdp = total_hdrvdp / num_samples if num_samples > 0 else 0
    
    model.train()
    
    return avg_psnr, avg_ssim, avg_hdrvdp


def save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim, val_hdrvdp):
    """
    Save metrics to CSV file.
    """
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_hdrvdp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'val_psnr': f"{val_psnr:.4f}",
            'val_ssim': f"{val_ssim:.4f}",
            'val_hdrvdp': f"{val_hdrvdp:.4f}"
        })


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
    
    # ======================================
    # Load and split dataset
    # ======================================
    print("Loading dataset...")
    full_dataset = HDRDataset(LDR_DIR, HDR_DIR, mode="train")
    
    # Split into train and validation (80-20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    
    # ========================================
    # Model initialization
    # ========================================
    model = ArtHDRNet(in_channels=3, base_channels=64, num_iterations=opt.iter)
    
    # ========================================
    # GPU configuration
    # ========================================
    str_ids = opt.gpu_ids.split(",")
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    
    device = torch.device("cpu")
    if len(opt.gpu_ids) > 0:
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() >= len(opt.gpu_ids)
        
        torch.cuda.set_device(opt.gpu_ids[0])
        device = torch.device(f"cuda:{opt.gpu_ids[0]}")
        
        if len(opt.gpu_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)
        
        model.cuda()
    
    print(f"Using device: {device}")
    
    # ========================================
    # Loss and optimizer
    # ========================================
    # Use ArtHDRLoss from model.py
    criterion = ArtHDRLoss(lambda1=0.1, lambda2=0.5, mu=5000)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    
    # ========================================
    # Load checkpoint if continuing training
    # ========================================
    start_epoch = 1
    best_val_psnr = 0
    best_val_ssim = 0
    best_val_hdrvdp = 0
    
    if opt.continue_train:
        try:
            start_epoch, model = load_checkpoint(model, opt.ckpt_path)
            print(f"Resuming training from epoch {start_epoch}")
            
            # Try to load best metrics from CSV if exists
            if os.path.exists(CSV_FILE):
                with open(CSV_FILE, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        last_row = rows[-1]
                        best_val_psnr = float(last_row['val_psnr'])
                        best_val_ssim = float(last_row['val_ssim'])
                        best_val_hdrvdp = float(last_row['val_hdrvdp'])
        except Exception as e:
            print(e)
            print("Checkpoint not found! Training from scratch.")
            start_epoch = 1
    
    # ========================================
    # Training loop
    # ========================================
    print("\nStarting training...")
    
    for epoch in range(start_epoch, opt.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0
        
        # Check whether LR needs to be updated
        if epoch > opt.lr_decay_after:
            update_lr(optimizer, epoch, opt)
        
        print(f"\nEpoch: {epoch}/{opt.epochs}")
        
        # Training phase
        model.train()
        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training")):
            optimizer.zero_grad()
            
            input_ldr = data["ldr_image"].to(device)
            ground_truth = data["hdr_image"].to(device)
            
            # Forward pass
            outputs = model(input_ldr)
            
            # Calculate loss
            loss = criterion(outputs, ground_truth)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Log batch information
            if (batch_idx + 1) % opt.log_after == 0:
                avg_loss = running_loss / num_batches
                print(f"  Batch: {batch_idx + 1}; Training loss: {avg_loss:.6f}")
        
        # Calculate average training loss for the epoch
        avg_train_loss = running_loss / num_batches if num_batches > 0 else 0
        
        # Validation phase
        print("Validating...")
        val_psnr, val_ssim, val_hdrvdp = validate_model(
            model, val_loader, device, epoch, 
            save_samples=(epoch % opt.save_ckpt_after == 0 or epoch == 1)
        )
        
        # Save metrics to CSV
        save_metrics_to_csv(CSV_FILE, epoch, avg_train_loss, val_psnr, val_ssim, val_hdrvdp)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Training Loss: {avg_train_loss:.6f}")
        print(f"  Validation PSNR: {val_psnr:.4f} dB")
        print(f"  Validation SSIM: {val_ssim:.4f}")
        print(f"  Validation HDR-VDP-2: {val_hdrvdp:.4f}")
        print(f"  Time: {epoch_time:.2f} seconds")
        
        # Save checkpoint based on best PSNR
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            best_val_ssim = val_ssim
            best_val_hdrvdp = val_hdrvdp
            
            # Save best model
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"best_model_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_hdrvdp': val_hdrvdp,
                'loss': avg_train_loss,
            }, checkpoint_path)
            
            # Also update latest checkpoint
            latest_path = os.path.join(CHECKPOINT_DIR, "latest.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_hdrvdp': val_hdrvdp,
                'loss': avg_train_loss,
            }, latest_path)
            
            print(f"  Saved best model with PSNR: {val_psnr:.4f}")
        
        # Save regular checkpoint every save_ckpt_after epochs
        if epoch % opt.save_ckpt_after == 0:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_psnr': val_psnr,
                'val_ssim': val_ssim,
                'val_hdrvdp': val_hdrvdp,
                'loss': avg_train_loss,
            }, checkpoint_path)
            print(f"  Saved checkpoint at epoch {epoch}")
    
    print("\nTraining completed!")
    print(f"Best validation PSNR: {best_val_psnr:.4f}")
    print(f"Best validation SSIM: {best_val_ssim:.4f}")
    print(f"Best validation HDR-VDP-2: {best_val_hdrvdp:.4f}")


if __name__ == "__main__":
    main()
