"""
ArtHDR-Net Training - DUAL GPU OPTIMIZED
Utilizes both RTX 4000 Ada (20GB) and RTX 3060 (12GB)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from model import ArtHDRNet, ArtHDRLoss, init_weights

# ==================== CONFIGURATION ====================
class Config:
    # Paths - MODIFY THESE
    LDR_DIR = "/path/to/your/ldr/images"
    HDR_DIR = "/path/to/your/hdr/images"
    CHECKPOINT_DIR = "checkpoint"
    CSV_FILE = "training_metrics.csv"
    
    # ========== OPTIMIZED FOR YOUR DUAL GPU SETUP ==========
    # RTX 4000 Ada (20GB) + RTX 3060 (12GB) = 32GB total
    BATCH_SIZE = 8  # Per GPU, so 8*2 = 16 total (better than paper's 10!)
    ACCUMULATION_STEPS = 1  # No need with large batch
    NUM_EPOCHS = 150
    LEARNING_RATE = 2e-4
    LR_DECAY_FACTOR = 0.5
    LR_DECAY_EPOCHS = [50, 100]
    IMAGE_SIZE = 512
    NUM_ITERATIONS = 4
    
    # Model parameters
    IN_CHANNELS = 3
    BASE_CHANNELS = 64
    
    # Loss parameters
    LAMBDA1 = 0.1
    LAMBDA2 = 0.5
    MU = 5000
    
    # Multi-GPU settings
    USE_MULTI_GPU = True  # Enable DataParallel
    GPU_IDS = [0, 1]  # Both GPUs
    PRIMARY_GPU = 0  # RTX 4000 Ada as primary
    
    # Training optimizations
    USE_MIXED_PRECISION = True
    SAVE_EVERY = 10
    
    # Device
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 8  # 4 per GPU, you have plenty of CPU cores

config = Config()

# ==================== DATASET ====================
class HDRDataset(Dataset):
    def __init__(self, ldr_dir, hdr_dir, image_size=512):
        self.ldr_dir = ldr_dir
        self.hdr_dir = hdr_dir
        self.image_size = image_size
        
        self.ldr_images = sorted(glob.glob(os.path.join(ldr_dir, "*.jpg")))
        if len(self.ldr_images) == 0:
            self.ldr_images = sorted(glob.glob(os.path.join(ldr_dir, "*.png")))
        
        print(f"Found {len(self.ldr_images)} LDR images")
        
        self.valid_pairs = []
        for ldr_path in self.ldr_images:
            basename = os.path.splitext(os.path.basename(ldr_path))[0]
            hdr_path = os.path.join(hdr_dir, basename + ".hdr")
            if os.path.exists(hdr_path):
                self.valid_pairs.append((ldr_path, hdr_path))
        
        print(f"Found {len(self.valid_pairs)} valid LDR-HDR pairs")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid pairs found!")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        ldr_path, hdr_path = self.valid_pairs[idx]
        
        ldr = cv2.imread(ldr_path)
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
        ldr = cv2.resize(ldr, (self.image_size, self.image_size))
        ldr = ldr.astype(np.float32) / 255.0
        
        hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdr is None:
            raise ValueError(f"Failed to load: {hdr_path}")
        hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        hdr = cv2.resize(hdr, (self.image_size, self.image_size))
        hdr = hdr.astype(np.float32)
        hdr = np.clip(hdr, 0, None)
        
        ldr = torch.from_numpy(ldr).permute(2, 0, 1).float()
        hdr = torch.from_numpy(hdr).permute(2, 0, 1).float()
        
        return ldr, hdr

# ==================== METRICS ====================
def calculate_psnr(img1, img2, max_value=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11, max_value=1.0):
    C1 = (0.01 * max_value) ** 2
    C2 = (0.03 * max_value) ** 2
    
    mu1 = torch.nn.functional.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = torch.nn.functional.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = torch.nn.functional.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = torch.nn.functional.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()

def mu_law_compression(hdr, mu=5000):
    return torch.log(1 + mu * hdr) / np.log(1 + mu)

# ==================== TRAINING ====================
def train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, device):
    model.train()
    epoch_loss = 0.0
    epoch_psnr = 0.0
    epoch_ssim = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [TRAIN]")
    
    for batch_idx, (ldr, hdr_gt) in enumerate(pbar):
        ldr = ldr.to(device)
        hdr_gt = hdr_gt.to(device)
        
        optimizer.zero_grad()
        
        if config.USE_MIXED_PRECISION:
            with autocast():
                hdr_outputs = model(ldr)
                loss = criterion(hdr_outputs, hdr_gt)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            hdr_outputs = model(ldr)
            loss = criterion(hdr_outputs, hdr_gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        with torch.no_grad():
            hdr_pred = hdr_outputs[-1]
            hdr_pred_tm = mu_law_compression(torch.clamp(hdr_pred, 0, None))
            hdr_gt_tm = mu_law_compression(hdr_gt)
            batch_psnr = calculate_psnr(hdr_pred_tm, hdr_gt_tm)
            batch_ssim = calculate_ssim(torch.clamp(hdr_pred, 0, 1), torch.clamp(hdr_gt, 0, 1))
        
        epoch_loss += loss.item()
        epoch_psnr += batch_psnr
        epoch_ssim += batch_ssim
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{batch_psnr:.2f}',
            'ssim': f'{batch_ssim:.4f}'
        })
    
    return epoch_loss / len(train_loader), epoch_psnr / len(train_loader), epoch_ssim / len(train_loader)

def validate_epoch(model, val_loader, criterion, epoch, device):
    model.eval()
    epoch_loss = 0.0
    epoch_psnr = 0.0
    epoch_ssim = 0.0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [VAL]")
    
    with torch.no_grad():
        for ldr, hdr_gt in pbar:
            ldr = ldr.to(device)
            hdr_gt = hdr_gt.to(device)
            
            if config.USE_MIXED_PRECISION:
                with autocast():
                    hdr_outputs = model(ldr)
                    loss = criterion(hdr_outputs, hdr_gt)
            else:
                hdr_outputs = model(ldr)
                loss = criterion(hdr_outputs, hdr_gt)
            
            hdr_pred = hdr_outputs[-1]
            hdr_pred_tm = mu_law_compression(torch.clamp(hdr_pred, 0, None))
            hdr_gt_tm = mu_law_compression(hdr_gt)
            batch_psnr = calculate_psnr(hdr_pred_tm, hdr_gt_tm)
            batch_ssim = calculate_ssim(torch.clamp(hdr_pred, 0, 1), torch.clamp(hdr_gt, 0, 1))
            
            epoch_loss += loss.item()
            epoch_psnr += batch_psnr
            epoch_ssim += batch_ssim
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{batch_psnr:.2f}',
                'ssim': f'{batch_ssim:.4f}'
            })
    
    return epoch_loss / len(val_loader), epoch_psnr / len(val_loader), epoch_ssim / len(val_loader)

# ==================== MAIN ====================
def main():
    print("="*80)
    print("ArtHDR-Net Training - DUAL GPU OPTIMIZED")
    print("="*80)
    
    # Check GPUs
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    print(f"\nBatch Size per GPU: {config.BATCH_SIZE}")
    print(f"Number of GPUs: {len(config.GPU_IDS)}")
    print(f"Total Batch Size: {config.BATCH_SIZE * len(config.GPU_IDS)}")
    print(f"Mixed Precision: {config.USE_MIXED_PRECISION}")
    print(f"Num Workers: {config.NUM_WORKERS}")
    print("="*80)
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    
    # Dataset
    full_dataset = HDRDataset(config.LDR_DIR, config.HDR_DIR, config.IMAGE_SIZE)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True, persistent_workers=True
    )
    
    # Model
    print("\nInitializing model...")
    model = ArtHDRNet(
        in_channels=config.IN_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        num_iterations=config.NUM_ITERATIONS
    )
    
    # Multi-GPU setup
    if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        print(f"Using DataParallel on GPUs: {config.GPU_IDS}")
        model = nn.DataParallel(model, device_ids=config.GPU_IDS)
        model = model.to(config.DEVICE)
    else:
        model = model.to(config.DEVICE)
        model.apply(init_weights)
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model parameters: {param_count:.2f}M")
    
    # Loss (needs to be on all GPUs via DataParallel wrapper)
    criterion = ArtHDRLoss(config.LAMBDA1, config.LAMBDA2, config.MU)
    if config.USE_MULTI_GPU and torch.cuda.device_count() > 1:
        criterion = nn.DataParallel(criterion, device_ids=config.GPU_IDS)
    criterion = criterion.to(config.DEVICE)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.LR_DECAY_EPOCHS, gamma=config.LR_DECAY_FACTOR
    )
    
    scaler = GradScaler() if config.USE_MIXED_PRECISION else None
    
    # CSV logging
    csv_file = open(config.CSV_FILE, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'Epoch', 'Train_Loss', 'Train_PSNR', 'Train_SSIM',
        'Val_Loss', 'Val_PSNR', 'Val_SSIM', 'Learning_Rate', 'Timestamp'
    ])
    
    best_val_psnr = -float('inf')
    best_epoch = 0
    checkpoint_epoch = 0
    checkpoint_psnr = -float('inf')
    
    print("\nStarting Training...\n")
    
    try:
        for epoch in range(config.NUM_EPOCHS):
            train_loss, train_psnr, train_ssim = train_epoch(
                model, train_loader, criterion, optimizer, scaler, epoch, config.DEVICE
            )
            
            val_loss, val_psnr, val_ssim = validate_epoch(
                model, val_loader, criterion, epoch, config.DEVICE
            )
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([
                epoch + 1, train_loss, train_psnr, train_ssim,
                val_loss, val_psnr, val_ssim, current_lr, timestamp
            ])
            csv_file.flush()
            
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
            print(f"  Train -> Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
            print(f"  Val   -> Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_epoch = epoch + 1
                print(f"  ★ New best PSNR: {best_val_psnr:.2f}")
            
            # Save checkpoint
            if (epoch + 1) % config.SAVE_EVERY == 0:
                should_save = checkpoint_epoch == 0 or val_psnr > checkpoint_psnr
                
                if should_save:
                    checkpoint_path = os.path.join(
                        config.CHECKPOINT_DIR, f'arthdrnet_epoch_{epoch+1}.pth'
                    )
                    
                    # Save model (handle DataParallel wrapper)
                    model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                    
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_psnr': val_psnr,
                        'best_val_psnr': best_val_psnr,
                        'best_epoch': best_epoch,
                    }, checkpoint_path)
                    
                    print(f"  💾 Checkpoint saved: {checkpoint_path}")
                    checkpoint_epoch = epoch + 1
                    checkpoint_psnr = val_psnr
                else:
                    print(f"  ⏭️  No improvement from epoch {checkpoint_epoch}")
            
            print("-"*80)
        
        # Final save
        final_path = os.path.join(config.CHECKPOINT_DIR, 'arthdrnet_final.pth')
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({
            'epoch': config.NUM_EPOCHS,
            'model_state_dict': model_state,
            'best_val_psnr': best_val_psnr,
            'best_epoch': best_epoch,
        }, final_path)
        
        print("\n" + "="*80)
        print("Training Completed!")
        print("="*80)
        print(f"Final model: {final_path}")
        print(f"Best PSNR: {best_val_psnr:.2f} at epoch {best_epoch}")
        print(f"Metrics saved: {config.CSV_FILE}")
        print("="*80)
    
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted!")
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save({'model_state_dict': model_state},
                   os.path.join(config.CHECKPOINT_DIR, 'interrupted.pth'))
        print("Model saved at interruption")
    
    finally:
        csv_file.close()

if __name__ == "__main__":
    main()

