import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
import numpy as np

class TransformationUnit(nn.Module):
    """
    Transformation Unit (TU) - Generates multi-exposed versions of input LDR image
    Uses simple exposure adjustment as described in the paper
    """
    def __init__(self):
        super(TransformationUnit, self).__init__()
    
    def forward(self, ldr_input):
        """
        Generate EV -2, 0, +2 versions from input LDR image
        Input: LDR image assumed to be at EV 0
        Output: Tuple of (ldr_ev_minus2, ldr_ev0, ldr_ev_plus2)
        """
        # EV 0 (original input)
        ldr_ev0 = ldr_input
        
        # EV -2: darken the image (divide by 4 in linear space)
        # Since we're working in sRGB, we approximate with power function
        ldr_ev_minus2 = torch.pow(ldr_input, 2.0)  # Approximation for darker exposure
        
        # EV +2: brighten the image (multiply by 4 in linear space)
        ldr_ev_plus2 = torch.pow(ldr_input, 0.5)  # Approximation for brighter exposure
        
        # Clamp to valid range [0, 1]
        ldr_ev_minus2 = torch.clamp(ldr_ev_minus2, 0.0, 1.0)
        ldr_ev_plus2 = torch.clamp(ldr_ev_plus2, 0.0, 1.0)
        
        return ldr_ev_minus2, ldr_ev0, ldr_ev_plus2

class FeatureUnit(nn.Module):
    """
    Feature Unit (FU) - Extracts features from multi-exposed LDR images
    Three parallel branches for EV -2, 0, +2
    Each branch has three 3x3 convolutional layers with ReLU activation
    """
    def __init__(self, in_channels=3, base_channels=64):
        super(FeatureUnit, self).__init__()
        self.base_channels = base_channels
        
        # Three parallel branches for different exposures
        self.branch_minus2 = self._make_branch(in_channels)
        self.branch_0 = self._make_branch(in_channels)
        self.branch_plus2 = self._make_branch(in_channels)
    
    def _make_branch(self, in_channels):
        """Create a feature extraction branch with three conv layers"""
        return nn.Sequential(
            nn.Conv2d(in_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.base_channels, self.base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, ldr_ev_minus2, ldr_ev0, ldr_ev_plus2):
        """
        Extract features from multi-exposed LDR images
        Returns individual features and combined feature map
        """
        # Extract features from each branch
        fe_minus2 = self.branch_minus2(ldr_ev_minus2)
        fe_0 = self.branch_0(ldr_ev0)
        fe_plus2 = self.branch_plus2(ldr_ev_plus2)
        
        # Combined feature map (element-wise addition)
        fe_all = fe_minus2 + fe_0 + fe_plus2
        
        # Return individual features for skip connections and combined features
        return fe_minus2, fe_0, fe_plus2, fe_all

class DilatedDenseBlock(nn.Module):
    """
    Dilated Dense Block for Feedback Unit
    Contains four 3x3 convolutional layers with dilation rate 3
    """
    def __init__(self, in_channels, growth_rate=32, dilation_rate=3):
        super(DilatedDenseBlock, self).__init__()
        self.growth_rate = growth_rate
        
        # Four dilated convolutional layers
        self.conv1 = nn.Conv2d(in_channels, growth_rate, 3, padding=dilation_rate, 
                              dilation=dilation_rate)
        self.conv2 = nn.Conv2d(in_channels + growth_rate, growth_rate, 3, 
                              padding=dilation_rate, dilation=dilation_rate)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_rate, growth_rate, 3, 
                              padding=dilation_rate, dilation=dilation_rate)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_rate, growth_rate, 3, 
                              padding=dilation_rate, dilation=dilation_rate)
        
        # 1x1 compression layers at beginning and end
        self.compression_in = nn.Conv2d(in_channels, in_channels, 1)
        self.compression_out = nn.Conv2d(in_channels + 4 * growth_rate, in_channels, 1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Input compression
        x_compressed = self.compression_in(x)
        
        # Dense connections with dilated convolutions
        out1 = self.relu(self.conv1(x_compressed))
        concat1 = torch.cat([x_compressed, out1], 1)
        
        out2 = self.relu(self.conv2(concat1))
        concat2 = torch.cat([concat1, out2], 1)
        
        out3 = self.relu(self.conv3(concat2))
        concat3 = torch.cat([concat2, out3], 1)
        
        out4 = self.relu(self.conv4(concat3))
        concat4 = torch.cat([concat3, out4], 1)
        
        # Output compression
        output = self.compression_out(concat4)
        
        return output

class FeedbackUnit(nn.Module):
    """
    Feedback Unit (FBU) - Processes features with feedback mechanism
    Contains three dilated dense blocks with global and local feedback
    """
    def __init__(self, in_channels=64, num_blocks=3, num_iterations=4):
        super(FeedbackUnit, self).__init__()
        self.num_iterations = num_iterations
        self.in_channels = in_channels
        
        # Initial compression layer
        self.initial_compression = nn.Conv2d(in_channels, in_channels, 1)
        
        # Three dilated dense blocks
        self.dilated_blocks = nn.ModuleList([
            DilatedDenseBlock(in_channels, growth_rate=32, dilation_rate=3)
            for _ in range(num_blocks)
        ])
        
        # Final 3x3 convolutional layer
        self.final_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Hidden state storage for feedback
        self.hidden_state = None
    
    def reset_hidden_state(self):
        """Reset hidden state for new sequence"""
        self.hidden_state = None
    
    def forward(self, fe_all):
        """
        Forward pass with feedback mechanism
        fe_all: Combined features from Feature Unit
        """
        # Initialize hidden state if first iteration
        if self.hidden_state is None:
            self.hidden_state = fe_all
        else:
            # Combine current input with previous hidden state
            self.hidden_state = fe_all + self.hidden_state
        
        # Initial compression
        x = self.initial_compression(self.hidden_state)
        
        # Pass through dilated dense blocks
        for block in self.dilated_blocks:
            x = block(x)
        
        # Final convolution
        fb_output = self.relu(self.final_conv(x))
        
        # Update hidden state for next iteration
        self.hidden_state = fb_output
        
        return fb_output

class ReconstructionUnit(nn.Module):
    """
    Reconstruction Unit (RU) - Reconstructs HDR image from features
    Three 3x3 convolutional layers with ReLU (first two) and TanH (last)
    """
    def __init__(self, in_channels=64, out_channels=3):
        super(ReconstructionUnit, self).__init__()
        
        self.reconstruction_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.Tanh()  # Output in range [-1, 1] for HDR
        )
    
    def forward(self, frs):
        """Reconstruct HDR image from final feature map"""
        return self.reconstruction_net(frs)

class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 network
    Based on ReLU activation layers as mentioned in the paper
    """
    def __init__(self):
        super(VGGPerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        features = vgg.features
        
        # Extract specific layers for perceptual loss (ReLU layers)
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        
        for x in range(2):  # relu1_1, relu1_2
            self.slice1.add_module(str(x), features[x])
        for x in range(2, 7):  # up to relu2_2
            self.slice2.add_module(str(x), features[x])
        for x in range(7, 12):  # up to relu3_4
            self.slice3.add_module(str(x), features[x])
        for x in range(12, 21):  # up to relu4_4
            self.slice4.add_module(str(x), features[x])
        
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x, y):
        """Compute perceptual loss between x and y"""
        x_features = []
        y_features = []
        
        # Extract features from each slice
        for slice_module in [self.slice1, self.slice2, self.slice3, self.slice4]:
            x = slice_module(x)
            y = slice_module(y)
            x_features.append(x)
            y_features.append(y)
        
        # Compute L1 loss between features
        perceptual_loss = 0
        for x_feat, y_feat in zip(x_features, y_features):
            perceptual_loss += F.l1_loss(x_feat, y_feat)
        
        return perceptual_loss

class ArtHDRNet(nn.Module):
    """
    Complete ArtHDR-Net architecture as described in the paper
    """
    def __init__(self, in_channels=3, base_channels=64, num_iterations=4):
        super(ArtHDRNet, self).__init__()
        self.num_iterations = num_iterations
        
        # Four main units
        self.transformation_unit = TransformationUnit()
        self.feature_unit = FeatureUnit(in_channels, base_channels)
        self.feedback_unit = FeedbackUnit(base_channels, num_iterations=num_iterations)
        self.reconstruction_unit = ReconstructionUnit(base_channels, in_channels)
        
        # Store features from middle branch for skip connections
        self.fe0_1 = None  # First conv layer features from EV0 branch
        self.fe0_2 = None  # Second conv layer features from EV0 branch
    
    def extract_mid_branch_features(self, ldr_ev0):
        """
        Extract low-level features from middle branch (EV0) for skip connections
        This is called during Feature Unit forward pass
        """
        # Pass through first two layers of EV0 branch
        x = self.feature_unit.branch_0[0](ldr_ev0)  # First conv
        x = self.feature_unit.branch_0[1](x)  # First ReLU
        self.fe0_1 = x
        
        x = self.feature_unit.branch_0[2](x)  # Second conv
        x = self.feature_unit.branch_0[3](x)  # Second ReLU
        self.fe0_2 = x
    
    def forward(self, ldr_input):
        """
        Complete forward pass of ArtHDR-Net
        Input: Single LDR image at EV0
        Output: List of HDR images at each iteration
        """
        # Reset feedback unit hidden state
        self.feedback_unit.reset_hidden_state()
        
        # Transformation Unit: Generate multi-exposed versions
        ldr_ev_minus2, ldr_ev0, ldr_ev_plus2 = self.transformation_unit(ldr_input)
        
        # Feature Unit: Extract features
        fe_minus2, fe_0, fe_plus2, fe_all = self.feature_unit(
            ldr_ev_minus2, ldr_ev0, ldr_ev_plus2
        )
        
        # Extract middle branch features for skip connections
        self.extract_mid_branch_features(ldr_ev0)
        
        # Store outputs for each iteration
        hdr_outputs = []
        
        # Feedback iterations
        for t in range(self.num_iterations):
            # Feedback Unit
            fb_t = self.feedback_unit(fe_all)
            
            # Combine with skip connections (Eq. 6)
            frs_t = self.fe0_1 + self.fe0_2 + fb_t
            
            # Reconstruction Unit
            hdr_t = self.reconstruction_unit(frs_t)
            hdr_outputs.append(hdr_t)
        
        return hdr_outputs

class ArtHDRLoss(nn.Module):
    """
    Combined loss function for ArtHDR-Net training
    Uses L1 loss and perceptual loss on tone-mapped HDR images
    """
    def __init__(self, lambda1=0.1, lambda2=0.5, mu=5000):
        super(ArtHDRLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu = mu
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = VGGPerceptualLoss()
    
    def mu_law_compression(self, hdr):
        """
        μ-law compression for tone mapping
        compresses hdr to ldr range for loss calculation
        """
        return torch.log(1 + self.mu * hdr) / torch.log(1 + torch.tensor(self.mu).to(hdr.device))
    
    def forward(self, hdr_outputs, hdr_gt):
        """
        Calculate loss between generated HDR and ground truth
        Applied to tone-mapped versions
        """
        total_loss = 0
        
        # Tone-map both generated and ground truth HDR
        hdr_gt_tm = self.mu_law_compression(hdr_gt)
        
        for t, hdr_pred in enumerate(hdr_outputs):
            hdr_pred_tm = self.mu_law_compression(hdr_pred)
            
            # L1 loss (Eq. 9)
            l1_loss = self.l1_loss(hdr_pred_tm, hdr_gt_tm)
            
            # Perceptual loss (Eq. 10)
            perceptual_loss = self.perceptual_loss(hdr_pred_tm, hdr_gt_tm)
            
            # Combined loss (Eq. 8)
            iteration_loss = self.lambda1 * l1_loss + self.lambda2 * perceptual_loss
            total_loss += iteration_loss
        
        return total_loss

# Utility function to initialize weights
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Example usage and training setup
if __name__ == "__main__":
    # Create model
    model = ArtHDRNet(in_channels=3, base_channels=64, num_iterations=4)
    model.apply(init_weights)
    
    # Create loss function
    criterion = ArtHDRLoss(lambda1=0.1, lambda2=0.5)
    
    # Example input
    batch_size, channels, height, width = 4, 3, 512, 512
    ldr_input = torch.randn(batch_size, channels, height, width)
    hdr_gt = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    hdr_outputs = model(ldr_input)
    
    # Calculate loss
    loss = criterion(hdr_outputs, hdr_gt)
    
    print(f"Model output: {len(hdr_outputs)} iterations")
    print(f"Final HDR shape: {hdr_outputs[-1].shape}")
    print(f"Total loss: {loss.item()}")
