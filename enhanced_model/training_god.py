"""
training.py - Fixed version without NaN issues
"""
import os
import time
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import Dynamic_attention_model
from losses import EnhancedModelLoss
from options import Options
from data_loader import HDRDataset
from image_transforms import LDRTransforms
from skimage.metrics import structural_similarity as compare_ssim
import torch.nn.functional as F

# ============ FIXED HELPER FUNCTIONS ============

def mu_tonemap(hdr_image, mu=5000.0):
    """
    Fixed mu-law tonemapping - NO NaN issues
    """
    # Clamp to [0, 1] range first
    hdr_normalized = torch.clamp((hdr_image + 1.0) / 2.0, 0.0, 1.0)
    
    # Log with numerical stability
    numerator = torch.log1p(mu * hdr_normalized)  # log1p(x) = log(1+x), more stable
    denominator = np.log(1.0 + mu)  # Constant, computed once
    
    return numerator / denominator

def compute_psnr_ssim(pred, gt):
    """Compute PSNR and SSIM"""
    pred_batch = pred.unsqueeze(0)
    gt_batch = gt.unsqueeze(0)
    
    # PSNR calculation
    mu_tonemap_gt = mu_tonemap(gt_batch)
    mu_tonemap_pred = mu_tonemap(pred_batch)
    mse = F.mse_loss(mu_tonemap_pred, mu_tonemap_gt)
    psnr = 10 * np.log10(1.0 / (mse.item() + 1e-10))
    
    # SSIM calculation
    generated = np.clip((pred.cpu().numpy().transpose(1, 2, 0) + 1) / 2.0, 0, 1)
    real = np.clip((gt.cpu().numpy().transpose(1, 2, 0) + 1) / 2.0, 0, 1)
    ssim = compare_ssim(generated, real, channel_axis=-1, win_size=7, data_range=1.0)
    
    return psnr, ssim

def validate_model(model, val_loader, device, epoch):
    """Simplified validation"""
    model.eval()
    ldr_transformer = LDRTransforms()
    
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            input_ldr = data["ldr_image"].to(device)
            ground_truth = data["hdr_image"].to(device)
            
            # Apply transformations
            original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)
            
            # Forward pass
            outputs = model(underexposed, overexposed, original)
            
            # Calculate metrics
            for i in range(outputs.shape[0]):
                psnr, ssim = compute_psnr_ssim(outputs[i], ground_truth[i])
                total_psnr += psnr
                total_ssim += ssim
                num_samples += 1
    
    model.train()
    avg_psnr = total_psnr / max(num_samples, 1)
    avg_ssim = total_ssim / max(num_samples, 1)
    
    return avg_psnr, avg_ssim

def save_metrics_to_csv(csv_path, epoch, train_loss, val_psnr, val_ssim):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'val_psnr', 'val_ssim']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'val_psnr': f"{val_psnr:.4f}",
            'val_ssim': f"{val_ssim:.4f}"
        })

def main():
    opt = Options().parse()
    
    # Dataset paths
    LDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"
    HDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"
    CHECKPOINT_DIR = "checkpoints"
    CSV_FILE = "training_metrics.csv"
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("./validation_results", exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    full_dataset = HDRDataset(LDR_DIR, HDR_DIR, mode="train")
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model (NO DeepSpeed, NO FP16)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Dynamic_attention_model(256, 512, 1024, 2048).to(device)
    
    # Use regular FP32 optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )
    
    # FIXED Loss function
    criterion = EnhancedModelLoss().to(device)
    
    print(f"\n✓ Model loaded on {device} in FP32 (no mixed precision)")
    print("✓ DeepSpeed REMOVED")
    
    # Load checkpoint if continuing
    start_epoch = 1
    best_val_psnr = 0
    
    if opt.continue_train:
        try:
            checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_psnr = checkpoint.get('best_val_psnr', 0)
            print(f"Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"No checkpoint found: {e}. Training from scratch.")
    
    # Training loop
    print("\nStarting training in FP32...")
    ldr_transformer = LDRTransforms()
    
    for epoch in range(start_epoch, opt.epochs + 1):
        epoch_start = time.time()
        model.train()
        
        running_loss = 0.0
        num_batches = 0
        
        # LR decay
        if epoch > opt.lr_decay_after:
            lr_scale = 1.0 - max(0, epoch - opt.lr_decay_after) / (opt.epochs - opt.lr_decay_after)
            for param_group in optimizer.param_groups:
                param_group['lr'] = opt.lr * lr_scale
        
        print(f"\nEpoch: {epoch}/{opt.epochs}")
        
        for batch_idx, data in enumerate(tqdm(train_loader, desc="Training")):
            input_ldr = data["ldr_image"].to(device)
            ground_truth = data["hdr_image"].to(device)
            
            # Check for NaN in input data
            if torch.isnan(input_ldr).any() or torch.isnan(ground_truth).any():
                print(f"⚠ WARNING: NaN detected in batch {batch_idx}, skipping...")
                continue
            
            # Apply transformations
            original, gamma, underexposed, overexposed, hist_eq, clahe = ldr_transformer(input_ldr)
            
            # Forward pass (NO autocast - pure FP32)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(underexposed, overexposed, original)
            
            # Calculate loss
            loss_out = criterion(outputs, ground_truth)
            if isinstance(loss_out, (tuple, list)):
                loss = loss_out[0]
            elif isinstance(loss_out, dict):
                loss = loss_out.get("loss", loss_out.get("total", loss_out))
            else:
                loss = loss_out
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠ WARNING: NaN/Inf loss at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Log batch info
            if (batch_idx + 1) % opt.log_after == 0:
                avg_loss = running_loss / num_batches
                tqdm.write(f"  Batch: {batch_idx + 1}; Training loss: {avg_loss:.6f}")
        
        avg_train_loss = running_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start
        
        # Validation
        print("Validating...")
        val_psnr, val_ssim = validate_model(model, val_loader, device, epoch)
        
        # Save metrics
        save_metrics_to_csv(CSV_FILE, epoch, avg_train_loss, val_psnr, val_ssim)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{opt.epochs} Summary")
        print(f"{'='*60}")
        print(f"  Training Loss    : {avg_train_loss:.6f}")
        print(f"  Validation PSNR  : {val_psnr:.4f} dB")
        print(f"  Validation SSIM  : {val_ssim:.4f}")
        print(f"  Epoch Time       : {epoch_time:.2f} seconds")
        print(f"{'='*60}")
        
        # Save checkpoint
        if val_psnr > best_val_psnr:
            best_val_psnr = val_psnr
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_psnr': best_val_psnr,
                'val_psnr': val_psnr,
                'val_ssim': val_ssim
            }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  ✓ Saved best model with PSNR: {val_psnr:.4f}")
        
        if epoch % opt.save_ckpt_after == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(CHECKPOINT_DIR, f'epoch_{epoch}.pth'))
            print(f"  ✓ Saved checkpoint at epoch {epoch}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"  Best PSNR: {best_val_psnr:.4f} dB")
    print("="*60)

if __name__ == "__main__":
    main()

