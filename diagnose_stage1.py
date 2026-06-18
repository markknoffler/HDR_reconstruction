
import torch
from model.instructpix2pix_pretrained_finetuned_stage1 import TrainableTriGateInstructPix2PixStage1
from model.training_scripts.data_loader import TriGateHDRDataset
from model.training_scripts.common_training import compute_psnr_ssim
import os

def diagnose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use default paths from your previous session context if available, or dummy paths
    ldr_dir = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/LDR_in"
    hdr_dir = "/home/user/Desktop/Deep_learning_projects/Hrishav_sir_project/Hrishav_Sir_FHDR/SingleHDR_training_data/HDR-Real/HDR_gt"
    
    if not os.path.exists(ldr_dir):
        print(f"LDR dir not found: {ldr_dir}")
        return

    ds = TriGateHDRDataset(ldr_dir, hdr_dir, max_dim=512)
    print(f"Dataset size: {len(ds)}")

    model = TrainableTriGateInstructPix2PixStage1.from_pretrained(device=device)
    model.eval()

    batch = ds[0]
    ldr = batch["ldr_image"].unsqueeze(0).to(device)
    hdr = batch["hdr_image"].unsqueeze(0).to(device)

    print(f"LDR range: {ldr.min().item():.4f} to {ldr.max().item():.4f}")
    print(f"HDR range: {hdr.min().item():.4f} to {hdr.max().item():.4f}")

    # Test 1: VAE Reconstruction
    with torch.no_grad():
        hdr_vae = model._hdr_to_vae(hdr)
        latents = model._vae_encode(hdr_vae, sample=False)
        recon_hdr = model._vae_decode_latents(latents).clamp(-1.0, 1.0)
    
    psnr_vae, ssim_vae = compute_psnr_ssim(recon_hdr[0], hdr[0])
    print(f"VAE Reconstruction PSNR: {psnr_vae:.4f} dB, SSIM: {ssim_vae:.4f}")

    # Test 2: Dummy Prediction (Identity)
    # Mapping LDR [0, 1] to [-1, 1]
    dummy_pred = 2.0 * ldr - 1.0
    psnr_id, ssim_id = compute_psnr_ssim(dummy_pred[0], hdr[0])
    print(f"LDR-as-HDR Identity PSNR: {psnr_id:.4f} dB, SSIM: {ssim_id:.4f}")

if __name__ == "__main__":
    diagnose()
