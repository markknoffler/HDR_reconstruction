
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import numpy as np
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import model architecture
from model import ArtHDRNet, ArtHDRLoss

# Metrics imports
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips  # For perceptual metrics

# =============================================================================
# CONFIGURATION - MODIFY THESE PATHS
# =============================================================================
LDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"  # Directory containing LDR images (.jpg)
HDR_DIR = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"  # Directory containing HDR images (.hdr)
CHECKPOINT_DIR = "checkpoint"
CSV_FILE = "training_metrics.csv"

# Training hyperparameters
BATCH_SIZE = 1  # Reduced from 10 for GPU memory
NUM_EPOCHS = 150
LEARNING_RATE = 2e-4
LR_DECAY_FACTOR = 0.5
LR_DECAY_EPOCHS = [50, 100]
IMAGE_SIZE = 512
NUM_ITERATIONS = 4
NUM_WORKERS = 4

# Loss weights
LAMBDA1 = 0.1
LAMBDA2 = 0.5
MU = 5000

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================================
# DATASET CLASS
# =============================================================================
class HDRDataset(Dataset):
    """Dataset for LDR-HDR image pairs"""

    def __init__(self, ldr_dir, hdr_dir, image_size=512, augment=False):
        self.ldr_dir = Path(ldr_dir)
        self.hdr_dir = Path(hdr_dir)
        self.image_size = image_size
        self.augment = augment

        # Get list of LDR images
        self.ldr_images = sorted([f for f in os.listdir(ldr_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        # Filter to only include images with HDR counterparts
        self.valid_pairs = []
        for ldr_file in self.ldr_images:
            hdr_file = os.path.splitext(ldr_file)[0] + '.hdr'
            hdr_path = self.hdr_dir / hdr_file
            if hdr_path.exists():
                self.valid_pairs.append((ldr_file, hdr_file))

        print(f"Found {len(self.valid_pairs)} valid LDR-HDR pairs")

    def __len__(self):
        return len(self.valid_pairs)

    def load_ldr(self, path):
        """Load and preprocess LDR image"""
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        return img

    def load_hdr(self, path):
        """Load and preprocess HDR image"""
        img = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Failed to load HDR image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        # Normalize HDR to reasonable range
        img = np.clip(img, 0, 10)  # Clip extreme values
        return img

    def __getitem__(self, idx):
        ldr_file, hdr_file = self.valid_pairs[idx]

        # Load images
        ldr_path = self.ldr_dir / ldr_file
        hdr_path = self.hdr_dir / hdr_file

        ldr_img = self.load_ldr(ldr_path)
        hdr_img = self.load_hdr(hdr_path)

        # Data augmentation
        if self.augment and np.random.rand() > 0.5:
            # Random horizontal flip
            ldr_img = np.flip(ldr_img, axis=1).copy()
            hdr_img = np.flip(hdr_img, axis=1).copy()

        # Convert to tensor
        ldr_tensor = torch.from_numpy(ldr_img).permute(2, 0, 1)
        hdr_tensor = torch.from_numpy(hdr_img).permute(2, 0, 1)

        return ldr_tensor, hdr_tensor, ldr_file

# =============================================================================
# METRICS CALCULATION
# =============================================================================
def mu_law_tonemap(hdr, mu=5000):
    """Apply mu-law compression for tone mapping"""
    hdr_clamped = torch.clamp(hdr, 0, 10)
    return torch.log(1 + mu * hdr_clamped) / np.log(1 + mu)

def calculate_psnr_torch(pred, target, max_val=1.0):
    """Calculate PSNR"""
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def calculate_ssim_torch(pred, target):
    """Calculate SSIM using sklearn"""
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    ssim_vals = []
    for i in range(pred_np.shape[0]):
        # Calculate SSIM for each image in batch
        p = pred_np[i].transpose(1, 2, 0)
        t = target_np[i].transpose(1, 2, 0)
        ssim_val = ssim(t, p, data_range=1.0, channel_axis=2)
        ssim_vals.append(ssim_val)

    return np.mean(ssim_vals)

def calculate_hdr_vdp2(pred, target):
    """
    Placeholder for HDR-VDP-2 calculation
    Note: Full HDR-VDP-2 requires specialized library installation
    This is a simplified approximation using perceptual metrics
    """
    # In practice, you would use the HDR-VDP-2 MATLAB or Python implementation
    # For now, we use a perceptual distance metric as proxy
    try:
        # This is a placeholder - returns a normalized score
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Simple perceptual quality metric (0-100 scale like HDR-VDP-2)
        mse = np.mean((pred_np - target_np) ** 2)
        q_score = 100 * np.exp(-10 * mse)  # Approximate quality score
        return float(q_score)
    except:
        return 0.0

def calculate_mae(pred, target):
    """Calculate Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()

def calculate_rmse(pred, target):
    """Calculate Root Mean Squared Error"""
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================
def train_epoch(model, train_loader, criterion, optimizer, epoch, device):
    """Train for one epoch"""
    model.train()

    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_mae = 0.0
    running_rmse = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]")

    for batch_idx, (ldr, hdr_gt, _) in enumerate(pbar):
        ldr = ldr.to(device)
        hdr_gt = hdr_gt.to(device)

        # Forward pass
        optimizer.zero_grad()
        hdr_outputs = model(ldr)

        # Calculate loss
        loss = criterion(hdr_outputs, hdr_gt)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Calculate metrics on tone-mapped images
        with torch.no_grad():
            pred_tm = mu_law_tonemap(hdr_outputs[-1])
            gt_tm = mu_law_tonemap(hdr_gt)

            batch_psnr = calculate_psnr_torch(pred_tm, gt_tm).item()
            batch_ssim = calculate_ssim_torch(pred_tm, gt_tm)
            batch_mae = calculate_mae(pred_tm, gt_tm)
            batch_rmse = calculate_rmse(pred_tm, gt_tm)

        # Update running metrics
        running_loss += loss.item()
        running_psnr += batch_psnr
        running_ssim += batch_ssim
        running_mae += batch_mae
        running_rmse += batch_rmse

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'psnr': f'{batch_psnr:.2f}',
            'ssim': f'{batch_ssim:.4f}'
        })

        # Clear cache periodically to prevent OOM
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    num_batches = len(train_loader)
    avg_metrics = {
        'loss': running_loss / num_batches,
        'psnr': running_psnr / num_batches,
        'ssim': running_ssim / num_batches,
        'mae': running_mae / num_batches,
        'rmse': running_rmse / num_batches
    }

    return avg_metrics

def validate_epoch(model, val_loader, criterion, epoch, device):
    """Validate for one epoch"""
    model.eval()

    running_loss = 0.0
    running_psnr = 0.0
    running_ssim = 0.0
    running_hdr_vdp2 = 0.0
    running_mae = 0.0
    running_rmse = 0.0

    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Val]")

    with torch.no_grad():
        for ldr, hdr_gt, _ in pbar:
            ldr = ldr.to(device)
            hdr_gt = hdr_gt.to(device)

            # Forward pass
            hdr_outputs = model(ldr)

            # Calculate loss
            loss = criterion(hdr_outputs, hdr_gt)

            # Calculate metrics on tone-mapped images
            pred_tm = mu_law_tonemap(hdr_outputs[-1])
            gt_tm = mu_law_tonemap(hdr_gt)

            batch_psnr = calculate_psnr_torch(pred_tm, gt_tm).item()
            batch_ssim = calculate_ssim_torch(pred_tm, gt_tm)
            batch_hdr_vdp2 = calculate_hdr_vdp2(pred_tm, gt_tm)
            batch_mae = calculate_mae(pred_tm, gt_tm)
            batch_rmse = calculate_rmse(pred_tm, gt_tm)

            # Update running metrics
            running_loss += loss.item()
            running_psnr += batch_psnr
            running_ssim += batch_ssim
            running_hdr_vdp2 += batch_hdr_vdp2
            running_mae += batch_mae
            running_rmse += batch_rmse

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'psnr': f'{batch_psnr:.2f}',
                'ssim': f'{batch_ssim:.4f}'
            })

    num_batches = len(val_loader)
    avg_metrics = {
        'loss': running_loss / num_batches,
        'psnr': running_psnr / num_batches,
        'ssim': running_ssim / num_batches,
        'hdr_vdp2': running_hdr_vdp2 / num_batches,
        'mae': running_mae / num_batches,
        'rmse': running_rmse / num_batches
    }

    return avg_metrics

def save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }

    filename = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, filename)

    if is_best:
        best_filename = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(checkpoint, best_filename)
        print(f"✓ Saved best model at epoch {epoch}")

    return filename

def should_save_checkpoint(current_metrics, past_metrics, epoch):
    """
    Determine if current checkpoint should be saved
    Compares current epoch with epoch-10
    """
    if epoch < 10:
        return True  # Save all checkpoints for first 10 epochs

    if past_metrics is None:
        return True

    # Compare main metrics: PSNR, SSIM, HDR-VDP-2
    # Current is better if any metric improved significantly
    psnr_improved = current_metrics['psnr'] > past_metrics.get('psnr', 0)
    ssim_improved = current_metrics['ssim'] > past_metrics.get('ssim', 0)
    hdr_vdp2_improved = current_metrics.get('hdr_vdp2', 0) > past_metrics.get('hdr_vdp2', 0)

    # Composite score (weighted average)
    current_score = (current_metrics['psnr'] / 40.0 + 
                     current_metrics['ssim'] + 
                     current_metrics.get('hdr_vdp2', 0) / 100.0) / 3.0

    past_score = (past_metrics.get('psnr', 0) / 40.0 + 
                  past_metrics.get('ssim', 0) + 
                  past_metrics.get('hdr_vdp2', 0) / 100.0) / 3.0

    return current_score > past_score

# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================
def main():
    print("="*80)
    print("ArtHDR-Net Training Script")
    print("="*80)

    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"✓ Checkpoint directory: {CHECKPOINT_DIR}")

    # Initialize dataset
    print("\nLoading dataset...")
    full_dataset = HDRDataset(LDR_DIR, HDR_DIR, IMAGE_SIZE, augment=True)

    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Initialize model
    print("\nInitializing model...")
    model = ArtHDRNet(
        in_channels=3, 
        base_channels=64, 
        num_iterations=NUM_ITERATIONS
    ).to(DEVICE)

    # Initialize loss function
    criterion = ArtHDRLoss(lambda1=LAMBDA1, lambda2=LAMBDA2, mu=MU).to(DEVICE)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=LR_DECAY_EPOCHS, 
        gamma=LR_DECAY_FACTOR
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model parameters: {num_params:,}")
    print(f"✓ Device: {DEVICE}")

    # Initialize CSV file
    csv_headers = [
        'epoch', 'train_loss', 'train_psnr', 'train_ssim', 'train_mae', 'train_rmse',
        'val_loss', 'val_psnr', 'val_ssim', 'val_hdr_vdp2', 'val_mae', 'val_rmse',
        'learning_rate', 'timestamp'
    ]

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    print(f"✓ Metrics CSV: {CSV_FILE}")

    # Training history for checkpoint comparison
    metrics_history = {}
    best_val_psnr = 0.0

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        print("-" * 80)

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, epoch, DEVICE)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, epoch, DEVICE)

        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, PSNR: {train_metrics['psnr']:.2f}, SSIM: {train_metrics['ssim']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}, HDR-VDP-2: {val_metrics['hdr_vdp2']:.2f}")
        print(f"  Learning Rate: {current_lr:.2e}")

        # Save metrics to CSV
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics['loss'], train_metrics['psnr'], train_metrics['ssim'],
                train_metrics['mae'], train_metrics['rmse'],
                val_metrics['loss'], val_metrics['psnr'], val_metrics['ssim'],
                val_metrics['hdr_vdp2'], val_metrics['mae'], val_metrics['rmse'],
                current_lr,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])

        # Store metrics for this epoch
        metrics_history[epoch] = val_metrics

        # Check if this is the best model
        is_best = val_metrics['psnr'] > best_val_psnr
        if is_best:
            best_val_psnr = val_metrics['psnr']

        # Save checkpoint every 10 epochs (with comparison logic)
        if epoch % 10 == 0:
            past_epoch = epoch - 10
            past_metrics = metrics_history.get(past_epoch, None)

            should_save = should_save_checkpoint(val_metrics, past_metrics, epoch)

            if should_save:
                save_checkpoint(model, optimizer, epoch, val_metrics, CHECKPOINT_DIR, is_best=False)
                print(f"✓ Saved checkpoint at epoch {epoch} (improved from epoch {past_epoch})")
            else:
                print(f"✗ Skipped checkpoint at epoch {epoch} (no improvement from epoch {past_epoch})")

        # Always save best model
        if is_best:
            save_checkpoint(model, optimizer, epoch, val_metrics, CHECKPOINT_DIR, is_best=True)

        # Clear GPU cache
        torch.cuda.empty_cache()

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation PSNR: {best_val_psnr:.2f}")
    print(f"All metrics saved to: {CSV_FILE}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
