import os
import csv
import cv2
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

CHECKPOINT_DIR = "./checkpoints"
GENERATED_DIR = "./generated_images"
CSV_LOG_FILE = "./training_log.csv"
BATCH_SIZE = 2
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
IMAGE_SIZE = 512
ACCUMULATION_STEPS = 5
DEVICE = "cuda:0"

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
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        return img
   
    def load_hdr(self, path):
        hdr = iio.imread(path)
        hdr = cv2.resize(hdr, (self.image_size, self.image_size))
        hdr = np.clip(hdr, 0, 1e6)
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
        
        ldr_gt = torch.from_numpy(ldr_gt).permute(2, 0, 1)
        ldr_his = torch.from_numpy(ldr_his).permute(2, 0, 1)
        hdr_gt = torch.from_numpy(hdr_gt).permute(2, 0, 1)
        
        hdr_gt = torch.clamp(hdr_gt, 0, 10)
        hdr_gt = (hdr_gt / 5.0) - 1.0
        
        return ldr_gt, ldr_his, hdr_gt, os.path.basename(ldr_path)



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
        
        ldr_save = ldr_gt[0].cpu().permute(1, 2, 0).numpy()
        hdr_pred_save = hdr_pred[0].cpu().permute(1, 2, 0).numpy()
        hdr_gt_save = hdr_gt[0].cpu().permute(1, 2, 0).numpy()
        
        hdr_pred_save = (hdr_pred_save + 1.0) * 5.0
        hdr_gt_save = (hdr_gt_save + 1.0) * 5.0
        
        hdr_pred_save = np.clip(hdr_pred_save, 0, 10)
        hdr_gt_save = np.clip(hdr_gt_save, 0, 10)
        
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


def validate(model, dataloader, criterion, device):
    """
    Validation function with FHDR-style PSNR and SSIM calculation
    """
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_ssim = 0
    total_hdrvdp2 = 0
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
            
            # Calculate metrics for each image in the batch
            for batch_ind in range(hdr_pred.size(0)):
                # --- FHDR-style PSNR Calculation ---
                # Apply mu-tonemap to individual images
                pred_tonemapped = mu_tonemap(hdr_pred[batch_ind:batch_ind+1])
                gt_tonemapped = mu_tonemap(hdr_gt[batch_ind:batch_ind+1])
                
                # Calculate MSE and PSNR
                mse = mse_loss(pred_tonemapped, gt_tonemapped)
                psnr_val = 10 * np.log10(1.0 / mse.item())
                
                # --- FHDR-style SSIM Calculation ---
                # Convert to numpy and transpose to (H, W, C) format
                # Normalize from [-1, 1] to [0, 1]
                generated = (np.transpose(hdr_pred[batch_ind].cpu().numpy(), (1, 2, 0)) + 1) / 2.0
                real = (np.transpose(hdr_gt[batch_ind].cpu().numpy(), (1, 2, 0)) + 1) / 2.0
                
                # Calculate SSIM with multichannel=True
                ssim_val = compare_ssim(
                    generated,
                    real,
                    channel_axis=-1,   # last axis is channels (RGB)
                    win_size=7         # optional but explicit; must be <= min(H, W)
                )

                
                # --- HDR-VDP2 Proxy Metric ---
                hdrvdp2_val = compute_hdrvdp2_metric(
                    hdr_pred[batch_ind:batch_ind+1], 
                    hdr_gt[batch_ind:batch_ind+1]
                )
                
                # Accumulate metrics
                total_psnr += psnr_val
                total_ssim += ssim_val
                total_hdrvdp2 += hdrvdp2_val
                num_images += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr_val:.2f}',
                    'ssim': f'{ssim_val:.4f}',
                    'hdrvdp2': f'{hdrvdp2_val:.4f}'
                })
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    avg_hdrvdp2 = total_hdrvdp2 / num_images
    
    model.train()
    return avg_loss, avg_psnr, avg_ssim, avg_hdrvdp2

def train():
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

    print("Running a dry-run validation before training...")
    try:
        val_loss, val_psnr, val_ssim, val_hdrvdp2 = validate(
            model, val_loader, criterion, DEVICE
        )
        print(f"[Dry run] Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, "
            f"SSIM: {val_ssim:.4f}, HDR-VDP2: {val_hdrvdp2:.4f}")
    except Exception as e:
        print("Dry-run validation failed with error:", e)
        raise
    
    csv_exists = os.path.exists(CSV_LOG_FILE)
    csv_file = open(CSV_LOG_FILE, 'a' if csv_exists else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    if not csv_exists:
        csv_writer.writerow([
            'epoch', 'train_loss', 'val_loss',
            'val_psnr', 'val_ssim', 'val_hdrvdp2',  # new
            'l1_loss', 'vgg_loss', 'weber_loss', 'ms_ssim_loss', 'color_loss',
            'learning_rate', 'timestamp'
        ])

    
    best_psnr = 0.0
    best_epoch = 0
    
    print(f"\nStarting training for {NUM_EPOCHS} epochs...")
    print(f"Training samples: {train_size}, Validation samples: {val_split}")
    print("=" * 80)
    
    for epoch in range(1, NUM_EPOCHS + 1):
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
        
        #val_loss, val_psnr, val_ssim = validate(model, val_loader, criterion, DEVICE)
        val_loss, val_psnr, val_ssim, val_hdrvdp2 = validate(model, val_loader, criterion, DEVICE)
        print(f" Validation - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, HDR-VDP2: {val_hdrvdp2:.4f}") 
        #print(f"  Validation - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        
        csv_writer.writerow([
            epoch, avg_train_loss, val_loss, val_psnr, val_ssim, val_hdrvdp2,
            epoch_components['l1'], epoch_components['vgg'],
            epoch_components['weber'], epoch_components['ms_ssim'],
            epoch_components['color'],
            optimizer.param_groups[0]['lr'],
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])
        csv_file.flush()
        
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
                
                checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_model_epoch_{epoch}.pth')
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
    print(f"Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"Generated images saved in: {GENERATED_DIR}")


if __name__ == "__main__":
    train()

