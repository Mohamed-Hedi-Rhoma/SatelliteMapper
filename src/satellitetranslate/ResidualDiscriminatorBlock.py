import torch
import torch.nn as nn
import torch.nn.functional as F
from satellitetranslate.ConditionalInstanceNorm2d import ConditionalInstanceNorm2d

class ResidualDiscriminatorBlock(nn.Module):
    """
    Residual block for the discriminator with optional conditional normalization.
    
    Args:
        channels (int): Number of input and output channels
        stride (int): Stride for the first convolution (downsampling when > 1)
        angle_dim (int): Dimension of angular information
        use_cond_norm (bool): Whether to use conditional normalization
        norm_type (str): Type of normalization to use
    """
    def __init__(
        self,
        channels,
        stride=1,
        angle_dim=4,
        use_cond_norm=True,
        norm_type="instance",
        block_name="Unnamed"  # Added block name for debugging
    ):
        super(ResidualDiscriminatorBlock, self).__init__()
        
        self.block_name = block_name
        self.channels = channels
        self.stride = stride
        
        # Print initialization info
        print(f"Creating {block_name} ResidualDiscriminatorBlock:")
        print(f"  Channels: {channels}, Stride: {stride}")
        print(f"  Conditional Norm: {use_cond_norm}, Norm Type: {norm_type}")
        
        # Main branch
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, bias=not use_cond_norm)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=not use_cond_norm)
        
        # Normalization
        self.use_cond_norm = use_cond_norm
        if use_cond_norm:
            self.norm1 = ConditionalInstanceNorm2d(channels, angle_dim)
            self.norm2 = ConditionalInstanceNorm2d(channels, angle_dim)
        else:
            if norm_type == "instance":
                self.norm1 = nn.InstanceNorm2d(channels)
                self.norm2 = nn.InstanceNorm2d(channels)
            elif norm_type == "batch":
                self.norm1 = nn.BatchNorm2d(channels)
                self.norm2 = nn.BatchNorm2d(channels)
            else:
                self.norm1 = nn.Identity()
                self.norm2 = nn.Identity()
        
        # Activation
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Shortcut connection if stride > 1 (for downsampling)
        self.use_shortcut = stride > 1
        if self.use_shortcut:
            self.shortcut = nn.Conv2d(channels, channels, kernel_size=1, stride=stride, padding=0)
    
    def forward(self, x, angles=None):
        """Forward pass of the residual discriminator block."""
        # Print input shape
        print(f"{self.block_name} input shape: {x.shape}")
        
        # Save input for residual connection
        identity = x
        
        # First conv + norm + activation
        x = self.conv1(x)
        print(f"{self.block_name} after conv1: {x.shape}")
        
        if self.use_cond_norm and angles is not None:
            x = self.norm1(x, angles)
        else:
            x = self.norm1(x)
        x = self.activation(x)
        
        # Second conv + norm
        x = self.conv2(x)
        print(f"{self.block_name} after conv2: {x.shape}")
        
        if self.use_cond_norm and angles is not None:
            x = self.norm2(x, angles)
        else:
            x = self.norm2(x)
        
        # Apply shortcut for downsampling
        if self.use_shortcut:
            identity = self.shortcut(identity)
            print(f"{self.block_name} shortcut shape: {identity.shape}")
        
        # Add residual connection
        x = x + identity
        
        # Final activation
        x = self.activation(x)
        
        # Print output shape
        print(f"{self.block_name} output shape: {x.shape}")
        
        return x