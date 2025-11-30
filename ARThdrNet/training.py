"""
ArtHDR-Net Training Script
Based on the paper: "ArtHDR-Net: Perceptually Realistic and Accurate HDR Content Creation"
Training parameters from paper: batch_size=10, epochs=150, lr=2e-4, image_size=512x512
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import your model (ensure model.py is in same directory)
from model import ArtHDRNet, ArtHDRLoss, init_weights

# ==================== CONFIGURATION ====================
class Config:
    # Paths - MODIFY THESE TO YOUR DATASET LOCATIONS
    LDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"  # Directory containing LDR .jpg images
    HDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"  # Directory containing HDR .hdr images
    CHECKPOINT_DIR = "checkpoint"  # Will be created automatically
    CSV_FILE = "training_metrics.csv"
    
    # Training parameters from paper
    BATCH_SIZE = 10
    NUM_EPOCHS = 150
    LEARNING_RATE = 2e-4
    LR_DECAY_FACTOR = 0.5
    LR_DECAY_EPOCHS = [50, 100]  # Decay at these epochs
    IMAGE_SIZE = 512
    NUM_ITERATIONS = 4  # Feedback iterations
    
    # Model parameters
    IN_CHANNELS = 3
    BASE_CHANNELS = 64
    
    # Loss parameters from paper
    LAMBDA1 = 0.1  # Weight for L1 loss
    LAMBDA2 = 0.5  # Weight for perceptual loss
    MU = 5000  # μ-law compression parameter
    
    # Checkpoint settings
    SAVE_EVERY = 10  # Check every 10 epochs
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4

config = Config()

# ==================== DATASET ====================
class HDRDataset(Dataset):
    """
    Dataset for loading LDR-HDR image pairs
    LDR: .jpg format, HDR: .hdr format
    """
    def __init__(self, ldr_dir, hdr_dir, image_size=512, transform=None):
        self.ldr_dir = ldr_dir
        self.hdr_dir = hdr_dir
        self.image_size = image_size
        self.transform = transform
        
        # Get all LDR images
        self.ldr_images = sorted(glob.glob(os.path.join(ldr_dir, "*.jpg")))
        if len(self.ldr_images) == 0:
            self.ldr_images = sorted(glob.glob(os.path.join(ldr_dir, "*.png")))
        
        print(f"Found {len(self.ldr_images)} LDR images in {ldr_dir}")
        
        # Match with HDR images
        self.valid_pairs = []
        for ldr_path in self.ldr_images:
            basename = os.path.splitext(os.path.basename(ldr_path))[0]
            hdr_path = os.path.join(hdr_dir, basename + ".hdr")
            
            if os.path.exists(hdr_path):
                self.valid_pairs.append((ldr_path, hdr_path))
        
        print(f"Found {len(self.valid_pairs)} valid LDR-HDR pairs")
        
        if len(self.valid_pairs) == 0:
            raise ValueError("No valid LDR-HDR pairs found! Check your dataset paths and file names.")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        ldr_path, hdr_path = self.valid_pairs[idx]
        
        # Load LDR image (already in [0, 255] uint8)
        ldr = cv2.imread(ldr_path)
        ldr = cv2.cvtColor(ldr, cv2.COLOR_BGR2RGB)
        ldr = cv2.resize(ldr, (self.image_size, self.image_size))
        ldr = ldr.astype(np.float32) / 255.0  # Normalize to [0, 1]
        
        # Load HDR image (already in linear space)
        hdr = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        if hdr is None:
            raise ValueError(f"Failed to load HDR image: {hdr_path}")
        hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        hdr = cv2.resize(hdr, (self.image_size, self.image_size))
        hdr = hdr.astype(np.float32)
        
        # Ensure HDR is non-negative for mu-law compression
        hdr = np.clip(hdr, 0, None)
        
        # Convert to torch tensors (C, H, W)
        ldr = torch.from_numpy(ldr).permute(2, 0, 1).float()
        hdr = torch.from_numpy(hdr).permute(2, 0, 1).float()
        
        return ldr, hdr

# ==================== METRICS ====================
def calculate_psnr(img1, img2, max_value=1.0):
    """
    Calculate PSNR between two images
    Images should be in range [0, max_value]
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_value / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(img1, img2, window_size=11, max_value=1.0):
    """
    Calculate SSIM between two images
    Simplified implementation for batch processing
    """
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
    """
    μ-law compression for tone mapping (for PSNR calculation)
    """
    return torch.log(1 + mu * hdr) / np.log(1 + mu)

# ==================== TRAINING FUNCTIONS ====================
def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    epoch_psnr = 0.0
    epoch_ssim = 0.0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [TRAIN]")
    
    for batch_idx, (ldr, hdr_gt) in enumerate(pbar):
        ldr = ldr.to(device)
        hdr_gt = hdr_gt.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        hdr_outputs = model(ldr)  # List of HDR outputs from each iteration
        
        # Calculate loss
        loss = criterion(hdr_outputs, hdr_gt)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate metrics on final iteration output
        with torch.no_grad():
            hdr_pred = hdr_outputs[-1]  # Last iteration output
            
            # PSNR on tone-mapped images (as per paper)
            hdr_pred_tm = mu_law_compression(torch.clamp(hdr_pred, 0, None))
            hdr_gt_tm = mu_law_compression(hdr_gt)
            batch_psnr = calculate_psnr(hdr_pred_tm, hdr_gt_tm)
            
            # SSIM on actual HDR (as per paper)
            batch_ssim = calculate_ssim(torch.clamp(hdr_pred, 0, 1), torch.clamp(hdr_gt, 0, 1))
        
        epoch_loss += loss.item()
        epoch_psnr += batch_psnr
        epoch_ssim += batch_ssim
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{batch_psnr:.2f}',
            'ssim': f'{batch_ssim:.4f}'
        })
    
    num_batches = len(train_loader)
    return epoch_loss / num_batches, epoch_psnr / num_batches, epoch_ssim / num_batches

def validate_epoch(model, val_loader, criterion, epoch, device):
    """Validate for one epoch"""
    model.eval()
    epoch_loss = 0.0
    epoch_psnr = 0.0
    epoch_ssim = 0.0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS} [VAL]")
    
    with torch.no_grad():
        for ldr, hdr_gt in pbar:
            ldr = ldr.to(device)
            hdr_gt = hdr_gt.to(device)
            
            # Forward pass
            hdr_outputs = model(ldr)
            
            # Calculate loss
            loss = criterion(hdr_outputs, hdr_gt)
            
            # Calculate metrics on final iteration output
            hdr_pred = hdr_outputs[-1]
            
            # PSNR on tone-mapped images
            hdr_pred_tm = mu_law_compression(torch.clamp(hdr_pred, 0, None))
            hdr_gt_tm = mu_law_compression(hdr_gt)
            batch_psnr = calculate_psnr(hdr_pred_tm, hdr_gt_tm)
            
            # SSIM on actual HDR
            batch_ssim = calculate_ssim(torch.clamp(hdr_pred, 0, 1), torch.clamp(hdr_gt, 0, 1))
            
            epoch_loss += loss.item()
            epoch_psnr += batch_psnr
            epoch_ssim += batch_ssim
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{batch_psnr:.2f}',
                'ssim': f'{batch_ssim:.4f}'
            })
    
    num_batches = len(val_loader)
    return epoch_loss / num_batches, epoch_psnr / num_batches, epoch_ssim / num_batches

# ==================== MAIN TRAINING LOOP ====================
def main():
    print("="*80)
    print("ArtHDR-Net Training Script")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"LDR Directory: {config.LDR_DIR}")
    print(f"HDR Directory: {config.HDR_DIR}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Number of Epochs: {config.NUM_EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Image Size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print("="*80)
    
    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    print(f"✓ Checkpoint directory created: {config.CHECKPOINT_DIR}")
    
    # Create dataset
    print("\nLoading dataset...")
    full_dataset = HDRDataset(
        ldr_dir=config.LDR_DIR,
        hdr_dir=config.HDR_DIR,
        image_size=config.IMAGE_SIZE
    )
    
    # Split dataset 80-20 as per paper
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    # Create model
    print("\nInitializing model...")
    model = ArtHDRNet(
        in_channels=config.IN_CHANNELS,
        base_channels=config.BASE_CHANNELS,
        num_iterations=config.NUM_ITERATIONS
    ).to(config.DEVICE)
    
    # Initialize weights
    model.apply(init_weights)
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Create loss function
    criterion = ArtHDRLoss(
        lambda1=config.LAMBDA1,
        lambda2=config.LAMBDA2,
        mu=config.MU
    ).to(config.DEVICE)
    print("✓ Loss function created")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    print(f"✓ Adam optimizer created with lr={config.LEARNING_RATE}")
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.LR_DECAY_EPOCHS,
        gamma=config.LR_DECAY_FACTOR
    )
    
    # CSV file for logging
    csv_file = open(config.CSV_FILE, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'Epoch', 'Train_Loss', 'Train_PSNR', 'Train_SSIM',
        'Val_Loss', 'Val_PSNR', 'Val_SSIM', 'Learning_Rate', 'Timestamp'
    ])
    print(f"✓ CSV logging file created: {config.CSV_FILE}")
    
    # Training tracking
    best_val_psnr = -float('inf')
    best_epoch = 0
    checkpoint_epoch = 0  # Track epoch for checkpoint comparison
    
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80 + "\n")
    
    try:
        for epoch in range(config.NUM_EPOCHS):
            # Train
            train_loss, train_psnr, train_ssim = train_epoch(
                model, train_loader, criterion, optimizer, epoch, config.DEVICE
            )
            
            # Validate
            val_loss, val_psnr, val_ssim = validate_epoch(
                model, val_loader, criterion, epoch, config.DEVICE
            )
            
            # Update learning rate
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            csv_writer.writerow([
                epoch + 1, train_loss, train_psnr, train_ssim,
                val_loss, val_psnr, val_ssim, current_lr, timestamp
            ])
            csv_file.flush()
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS} Summary:")
            print(f"  Train -> Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
            print(f"  Val   -> Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
            print(f"  LR: {current_lr:.6f}")
            
            # Track best model
            if val_psnr > best_val_psnr:
                best_val_psnr = val_psnr
                best_epoch = epoch + 1
                print(f"  ★ New best validation PSNR: {best_val_psnr:.2f} at epoch {best_epoch}")
            
            # Save checkpoint every 10 epochs (only if better than 10 epochs ago)
            if (epoch + 1) % config.SAVE_EVERY == 0:
                # Check if current performance is better than checkpoint_epoch
                should_save = False
                
                if checkpoint_epoch == 0:
                    # First checkpoint, always save
                    should_save = True
                    reason = "First checkpoint"
                elif val_psnr > checkpoint_psnr:
                    # Better than previous checkpoint
                    should_save = True
                    reason = f"Improved from {checkpoint_psnr:.2f} to {val_psnr:.2f}"
                else:
                    reason = f"No improvement from {checkpoint_psnr:.2f}"
                
                if should_save:
                    checkpoint_path = os.path.join(
                        config.CHECKPOINT_DIR,
                        f'arthdrnet_epoch_{epoch+1}.pth'
                    )
                    
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'train_psnr': train_psnr,
                        'train_ssim': train_ssim,
                        'val_loss': val_loss,
                        'val_psnr': val_psnr,
                        'val_ssim': val_ssim,
                        'best_val_psnr': best_val_psnr,
                        'best_epoch': best_epoch,
                    }, checkpoint_path)
                    
                    print(f"  💾 Checkpoint saved: {checkpoint_path}")
                    print(f"     Reason: {reason}")
                    
                    # Update checkpoint tracking
                    checkpoint_epoch = epoch + 1
                    checkpoint_psnr = val_psnr
                else:
                    print(f"  ⏭️  Checkpoint not saved: {reason}")
            
            print("-"*80)
        
        # Save final model
        final_model_path = os.path.join(config.CHECKPOINT_DIR, 'arthdrnet_final.pth')
        torch.save({
            'epoch': config.NUM_EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_psnr': best_val_psnr,
            'best_epoch': best_epoch,
        }, final_model_path)
        
        print("\n" + "="*80)
        print("Training Completed!")
        print("="*80)
        print(f"Final model saved: {final_model_path}")
        print(f"Best validation PSNR: {best_val_psnr:.2f} at epoch {best_epoch}")
        print(f"Training metrics saved to: {config.CSV_FILE}")
        print("="*80)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        interrupt_path = os.path.join(config.CHECKPOINT_DIR, 'arthdrnet_interrupted.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_psnr': best_val_psnr,
        }, interrupt_path)
        print(f"Model saved at interruption: {interrupt_path}")
    
    finally:
        csv_file.close()

if __name__ == "__main__":
    main()

