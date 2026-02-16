import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset



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
        ldr_img = ldr_img.astype(np.float32) / 255.0
        ldr_img = torch.from_numpy(ldr_img).permute(2, 0, 1)
        
        # Load HDR image - FIXED VERSION
        try:
            if hdr_path.endswith('.npy'):
                hdr_img = np.load(hdr_path)
            else:
                hdr_img = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
                if hdr_img is None:
                    raise ValueError(f"Failed to load HDR image: {hdr_path}")
                if hdr_img.shape[-1] == 3:
                    hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
            
            hdr_img = hdr_img.astype(np.float32)
            
            # FIXED: Better normalization
            hdr_img = np.clip(hdr_img, 0.0, None)  # Remove negative values
            maxv = np.percentile(hdr_img, 99)  # Use 99th percentile instead of max
            if maxv > 1e-6:  # Increased threshold
                hdr_img = np.clip(hdr_img / maxv, 0.0, 1.0)
            else:
                print(f"⚠ WARNING: HDR image {hdr_path} has very low values, normalizing to 0.5")
                hdr_img = np.ones_like(hdr_img) * 0.5  # Use 0.5 instead of 1.0
            
            hdr_img = 2.0 * hdr_img - 1.0  # Now in [-1, 1]
            
        except Exception as e:
            print(f"❌ ERROR loading HDR {hdr_path}: {e}")
            print("Skipping this sample")
            # Return next sample instead of dummy data
            return self.__getitem__((idx + 1) % len(self))
        
        hdr_img = torch.from_numpy(hdr_img).permute(2, 0, 1)
        
        return {
            "ldr_image": ldr_img,
            "hdr_image": hdr_img,
            "ldr_path": self.ldr_files[idx],
            "hdr_path": self.hdr_files[idx]
        }

#    def __getitem__(self, idx):
#        ldr_path = os.path.join(self.ldr_dir, self.ldr_files[idx])
#        hdr_path = os.path.join(self.hdr_dir, self.hdr_files[idx])
#        
#        # Load LDR image
#        ldr_img = cv2.imread(ldr_path)
#        if ldr_img is None:
#            raise ValueError(f"Failed to load LDR image: {ldr_path}")
#        
#        ldr_img = cv2.cvtColor(ldr_img, cv2.COLOR_BGR2RGB)
#        ldr_img = ldr_img.astype(np.float32) / 255.0  # [0, 1]
#        #ldr_img = 2.0 * ldr_img - 1.0  # [-1, 1]
#        ldr_img = torch.from_numpy(ldr_img).permute(2, 0, 1)  # HWC to CHW
#        
#        # Load HDR image
#        # For simplicity, we'll assume HDR images are stored as numpy arrays
#        # You may need to adjust this based on your actual HDR format
#        try:
#            # Try to load as numpy array
#            if hdr_path.endswith('.npy'):
#                hdr_img = np.load(hdr_path)
#            else:
#                # For .hdr files, you might need a custom loader
#                # Here's a simple placeholder - adjust as needed
#                hdr_img = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
#                if hdr_img is None:
#                    raise ValueError(f"Failed to load HDR image: {hdr_path}")
#                hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
#                hdr_img = hdr_img.astype(np.float32)
#
#
#        except:
#            # If loading fails, create a dummy HDR image
#            print(f"Warning: Could not load HDR image {hdr_path}, using dummy")
#            hdr_img = np.ones((ldr_img.shape[1], ldr_img.shape[2], 3), dtype=np.float32)
#        
#        hdr_img = hdr_img.astype(np.float32)
#        # Normalize HDR to [-1, 1] range
#        # This depends on your HDR data - adjust as needed
#        hdr_img = hdr_img.astype(np.float32)
#        hdr_img = np.clip(hdr_img, 0.0, None)
#        maxv = float(hdr_img.max())
#        if maxv > 0:
#            hdr_img = hdr_img / maxv
#        hdr_img = 2.0 * hdr_img - 1.0
#
#        hdr_img = torch.from_numpy(hdr_img).permute(2, 0, 1)  # HWC to CHW
#        
#        return {
#            "ldr_image": ldr_img,
#            "hdr_image": hdr_img,
#            "ldr_path": self.ldr_files[idx],
#            "hdr_path": self.hdr_files[idx]
#        }
#
#
#
