"""
image_transforms.py

Transforms a single LDR image into multiple representations for HDR reconstruction.
Generates 6 input streams for the Dynamic_attention_model:
1. Original LDR (EV 0)
2. Gamma-corrected
3. Underexposed (EV -2)
4. Overexposed (EV +2)
5. Histogram Equalized
6. CLAHE (Contrast Limited Adaptive Histogram Equalization)
"""

import torch
import numpy as np
import cv2


class LDRTransforms:
    """
    Class to generate multiple LDR representations from a single input image.
    Works with PyTorch tensors in the format (B, C, H, W) with values in [0, 1].
    """
    
    def __init__(self, 
                 gamma_value=2.2,
                 underexposed_ev=-2.0,
                 overexposed_ev=2.0,
                 clahe_clip_limit=2.0,
                 clahe_tile_size=8):
        """
        Initialize transform parameters.
        
        Args:
            gamma_value (float): Gamma correction value (default: 2.2)
            underexposed_ev (float): EV adjustment for underexposure (default: -2.0)
            overexposed_ev (float): EV adjustment for overexposure (default: +2.0)
            clahe_clip_limit (float): CLAHE clip limit (default: 2.0)
            clahe_tile_size (int): CLAHE tile grid size (default: 8)
        """
        self.gamma_value = gamma_value
        self.underexposed_ev = underexposed_ev
        self.overexposed_ev = overexposed_ev
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        
        # Initialize CLAHE object
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit,
            tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
        )
    
    def apply_gamma_correction(self, img_tensor):
        """
        Apply gamma correction to the image.
        
        Args:
            img_tensor (torch.Tensor): Input tensor (B, C, H, W) in [0, 1]
            
        Returns:
            torch.Tensor: Gamma-corrected tensor (B, C, H, W) in [0, 1]
        """
        # Gamma correction: out = in^(1/gamma)
        gamma_corrected = torch.pow(img_tensor.clamp(min=1e-8), 1.0 / self.gamma_value)
        return gamma_corrected.clamp(0, 1)
    
    def apply_exposure_adjustment(self, img_tensor, ev_adjustment):
        """
        Apply exposure value (EV) adjustment.
        EV adjustment formula: new_value = old_value * 2^(EV)
        
        Args:
            img_tensor (torch.Tensor): Input tensor (B, C, H, W) in [0, 1]
            ev_adjustment (float): EV adjustment value (positive=brighter, negative=darker)
            
        Returns:
            torch.Tensor: Exposure-adjusted tensor (B, C, H, W) in [0, 1]
        """
        # Convert EV to linear scale
        scale_factor = 2.0 ** ev_adjustment
        adjusted = img_tensor * scale_factor
        return adjusted.clamp(0, 1)
    
    def apply_histogram_equalization(self, img_tensor):
        """
        Apply histogram equalization to each image in the batch.
        Operates on luminance channel in YCrCb color space.
        
        Args:
            img_tensor (torch.Tensor): Input tensor (B, C, H, W) in [0, 1]
            
        Returns:
            torch.Tensor: Histogram-equalized tensor (B, C, H, W) in [0, 1]
        """
        batch_size = img_tensor.shape[0]
        device = img_tensor.device
        
        equalized_batch = []
        
        for i in range(batch_size):
            # Convert to numpy (C, H, W) -> (H, W, C)
            img_np = img_tensor[i].cpu().numpy().transpose(1, 2, 0)
            
            # Convert to uint8 for OpenCV
            img_uint8 = (img_np * 255).astype(np.uint8)
            
            # Convert to YCrCb color space
            img_ycrcb = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2YCrCb)
            
            # Apply histogram equalization to Y channel
            img_ycrcb[:, :, 0] = cv2.equalizeHist(img_ycrcb[:, :, 0])
            
            # Convert back to RGB
            img_eq = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)
            
            # Convert back to float tensor [0, 1]
            img_eq_float = img_eq.astype(np.float32) / 255.0
            
            # Convert to tensor (H, W, C) -> (C, H, W)
            img_eq_tensor = torch.from_numpy(img_eq_float.transpose(2, 0, 1))
            equalized_batch.append(img_eq_tensor)
        
        # Stack batch
        equalized_tensor = torch.stack(equalized_batch, dim=0).to(device)
        return equalized_tensor
    
    def apply_clahe(self, img_tensor):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Operates on luminance channel in LAB color space for better results.
        
        Args:
            img_tensor (torch.Tensor): Input tensor (B, C, H, W) in [0, 1]
            
        Returns:
            torch.Tensor: CLAHE-enhanced tensor (B, C, H, W) in [0, 1]
        """
        batch_size = img_tensor.shape[0]
        device = img_tensor.device
        
        clahe_batch = []
        
        for i in range(batch_size):
            # Convert to numpy (C, H, W) -> (H, W, C)
            img_np = img_tensor[i].cpu().numpy().transpose(1, 2, 0)
            
            # Convert to uint8 for OpenCV
            img_uint8 = (img_np * 255).astype(np.uint8)
            
            # Convert to LAB color space
            img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            img_lab[:, :, 0] = self.clahe.apply(img_lab[:, :, 0])
            
            # Convert back to RGB
            img_clahe = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            # Convert back to float tensor [0, 1]
            img_clahe_float = img_clahe.astype(np.float32) / 255.0
            
            # Convert to tensor (H, W, C) -> (C, H, W)
            img_clahe_tensor = torch.from_numpy(img_clahe_float.transpose(2, 0, 1))
            clahe_batch.append(img_clahe_tensor)
        
        # Stack batch
        clahe_tensor = torch.stack(clahe_batch, dim=0).to(device)
        return clahe_tensor
    
    def generate_all_transforms(self, ldr_original):
        """
        Generate all 6 LDR representations from a single input image.
        
        Args:
            ldr_original (torch.Tensor): Original LDR image tensor (B, C, H, W) in [0, 1]
            
        Returns:
            dict: Dictionary containing all 6 transformations:
                - 'original': Original LDR (B, C, H, W)
                - 'gamma': Gamma-corrected (B, C, H, W)
                - 'underexposed': Underexposed EV-2 (B, C, H, W)
                - 'overexposed': Overexposed EV+2 (B, C, H, W)
                - 'hist_eq': Histogram equalized (B, C, H, W)
                - 'clahe': CLAHE enhanced (B, C, H, W)
        """
        # Ensure input is in correct range
        ldr_original = ldr_original.clamp(0, 1)
        
        transforms = {
            'original': ldr_original,
            'gamma': self.apply_gamma_correction(ldr_original),
            'underexposed': self.apply_exposure_adjustment(ldr_original, self.underexposed_ev),
            'overexposed': self.apply_exposure_adjustment(ldr_original, self.overexposed_ev),
            'hist_eq': self.apply_histogram_equalization(ldr_original),
            'clahe': self.apply_clahe(ldr_original)
        }
        
        return transforms
    
    def __call__(self, ldr_original):
        """
        Convenience method to generate all transforms.
        
        Args:
            ldr_original (torch.Tensor): Original LDR image tensor (B, C, H, W) in [0, 1]
            
        Returns:
            tuple: (original, gamma, underexposed, overexposed, hist_eq, clahe)
        """
        transforms = self.generate_all_transforms(ldr_original)
        
        return (
            transforms['original'],
            transforms['gamma'],
            transforms['underexposed'],
            transforms['overexposed'],
            transforms['hist_eq'],
            transforms['clahe']
        )


# Convenience function for direct use
def generate_ldr_inputs(ldr_original, 
                       gamma_value=2.2,
                       underexposed_ev=-2.0,
                       overexposed_ev=2.0,
                       clahe_clip_limit=2.0,
                       clahe_tile_size=8):
    """
    Generate all 6 LDR representations from a single input image.
    
    Args:
        ldr_original (torch.Tensor): Original LDR image tensor (B, C, H, W) in [0, 1]
        gamma_value (float): Gamma correction value (default: 2.2)
        underexposed_ev (float): EV adjustment for underexposure (default: -2.0)
        overexposed_ev (float): EV adjustment for overexposure (default: +2.0)
        clahe_clip_limit (float): CLAHE clip limit (default: 2.0)
        clahe_tile_size (int): CLAHE tile grid size (default: 8)
        
    Returns:
        tuple: (original, gamma, underexposed, overexposed, hist_eq, clahe)
               Each element is a torch.Tensor of shape (B, C, H, W) in [0, 1]
    """
    transformer = LDRTransforms(
        gamma_value=gamma_value,
        underexposed_ev=underexposed_ev,
        overexposed_ev=overexposed_ev,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size
    )
    
    return transformer(ldr_original)


if __name__ == "__main__":
    # Test the transforms
    print("Testing LDR Transforms...")
    
    # Create a dummy input tensor (batch_size=2, channels=3, height=256, width=256)
    dummy_ldr = torch.rand(2, 3, 256, 256)
    
    # Initialize transformer
    transformer = LDRTransforms()
    
    # Generate all transforms
    original, gamma, underexposed, overexposed, hist_eq, clahe = transformer(dummy_ldr)
    
    print(f"Original shape: {original.shape}, range: [{original.min():.3f}, {original.max():.3f}]")
    print(f"Gamma shape: {gamma.shape}, range: [{gamma.min():.3f}, {gamma.max():.3f}]")
    print(f"Underexposed shape: {underexposed.shape}, range: [{underexposed.min():.3f}, {underexposed.max():.3f}]")
    print(f"Overexposed shape: {overexposed.shape}, range: [{overexposed.min():.3f}, {overexposed.max():.3f}]")
    print(f"Hist EQ shape: {hist_eq.shape}, range: [{hist_eq.min():.3f}, {hist_eq.max():.3f}]")
    print(f"CLAHE shape: {clahe.shape}, range: [{clahe.min():.3f}, {clahe.max():.3f}]")
    
    print("\nAll transforms generated successfully!")

