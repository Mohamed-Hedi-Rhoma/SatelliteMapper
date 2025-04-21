import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the project root directory to system path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import the ConditionalInstanceNorm2d class
try:
    from satellitetranslate.ConditionalInstanceNorm2d import ConditionalInstanceNorm2d
except ImportError:
    # Try a relative import if the above fails
    from ConditionalInstanceNorm2d import ConditionalInstanceNorm2d


class ResidualBlock(nn.Module):
    """
    Residual block with optional conditional normalization.
    
    This block consists of two convolutional layers with normalization and ReLU activations,
    plus a skip connection from input to output.
    
    Args:
        channels (int): Number of input/output channels
        angle_dim (int): Dimension of the angle vector (e.g., 4 for solar/view azimuth/zenith)
        use_cond_norm (bool): Whether to use conditional normalization
        norm_type (str): Type of normalization ('instance', 'batch', or 'none')
        kernel_size (int): Size of the convolutional kernel
    """
    def __init__(self, channels, angle_dim=4, use_cond_norm=True, norm_type='instance', kernel_size=3):
        super(ResidualBlock, self).__init__()
        self.use_cond_norm = use_cond_norm
        padding = kernel_size // 2  # Same padding
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            channels, 
            channels, 
            kernel_size=kernel_size, 
            padding=padding
        )
        
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        
        # Normalization layers
        if use_cond_norm and norm_type == 'instance':
            self.norm1 = ConditionalInstanceNorm2d(channels, angle_dim)
            self.norm2 = ConditionalInstanceNorm2d(channels, angle_dim)
        elif norm_type == 'instance':
            self.norm1 = nn.InstanceNorm2d(channels)
            self.norm2 = nn.InstanceNorm2d(channels)
        elif norm_type == 'batch':
            self.norm1 = nn.BatchNorm2d(channels)
            self.norm2 = nn.BatchNorm2d(channels)
        else:  # 'none'
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
        
    def forward(self, x, angles=None):
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input feature map of shape [B, C, H, W]
            angles (torch.Tensor, optional): Angle information of shape [B, angle_dim]
                Required if use_cond_norm=True.
                
        Returns:
            torch.Tensor: Output feature map of shape [B, C, H, W]
        """
        # Store the input for the skip connection
        identity = x
        
        # First convolutional layer
        out = self.conv1(x)
        
        # First normalization
        if self.use_cond_norm and angles is not None:
            out = self.norm1(out, angles)
        else:
            out = self.norm1(out)
            
        # ReLU activation
        out = self.relu(out)
        
        # Second convolutional layer
        out = self.conv2(out)
        
        # Second normalization
        if self.use_cond_norm and angles is not None:
            out = self.norm2(out, angles)
        else:
            out = self.norm2(out)
        
        # Skip connection and final activation
        out = out + identity
        out = self.relu(out)
        
        return out


# Example usage:
if __name__ == "__main__":
    # Create a residual block with conditional normalization
    block = ResidualBlock(
        channels=64,
        angle_dim=4,
        use_cond_norm=True,
        norm_type='instance'
    )
    
    # Create dummy input tensors
    features = torch.randn(2, 64, 32, 32)  # [batch_size, channels, height, width]
    angles = torch.randn(2, 4)             # [batch_size, angle_dim]
    
    # Forward pass
    output = block(features, angles)
    
    # Print shapes
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output should match input shape: {output.shape == features.shape}")