import torch
import torch.nn as nn
import torch.nn.functional as F



class DownsamplingBlock(nn.Module):
    """
    Downsampling block for decreasing spatial resolution.
    
    This block decreases the spatial dimensions of feature maps while maintaining
    or changing the number of channels. It supports different downsampling methods.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels=None, 
        scale_factor=3.0,  # 10m → 30m is 3x downsampling
        mode='conv',  # 'conv', 'avgpool', or 'adaptive'
        norm_type='instance', 
        use_activation=True
    ):
        super(DownsamplingBlock, self).__init__()
        
        # If out_channels is not specified, use the same as in_channels
        out_channels = out_channels or in_channels
        
        # Print initialization info for debugging
        print(f"Creating DownsamplingBlock: in_channels={in_channels}, out_channels={out_channels}, scale={scale_factor}")
        
        # Create the downsampling layers
        if mode == 'conv':
            # Strided convolution for downsampling
            stride = round(scale_factor)
            print(f"Using strided convolution with stride={stride}")
            self.downsample = nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size=4,
                stride=stride,
                padding=1
            )
        elif mode == 'avgpool':
            # Average pooling
            print(f"Using average pooling with stride={int(scale_factor)}")
            self.downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=3, 
                    stride=int(scale_factor),
                    padding=1
                ),
                # Add a convolution to change the number of channels
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=1
                )
            )
        elif mode == 'adaptive':
            # For non-integer downsampling factors
            print(f"Using adaptive pooling with scale_factor={scale_factor}")
            self.scale_factor = scale_factor
            self.downsample = nn.Sequential(
                # Conv to adjust channels
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                # Adaptive avg pooling will be applied in forward pass
            )
        
        # Add normalization layer
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        else:  # 'none'
            self.norm = nn.Identity()
        
        # Activation
        self.use_activation = use_activation
        self.mode = mode
        if use_activation:
            self.activation = nn.ReLU(inplace=True)
            
    def forward(self, x):
        """
        Forward pass of the downsampling block.
        """
        # Print input shape for debugging
        print(f"DownsamplingBlock input shape: {x.shape}")
        
        # For adaptive mode, calculate target size
        if self.mode == 'adaptive':
            batch_size, channels, height, width = x.shape
            new_height = int(height / self.scale_factor)
            new_width = int(width / self.scale_factor)
            
            # Apply convolution
            out = self.downsample[0](x)
            # Apply adaptive pooling
            out = F.adaptive_avg_pool2d(out, (new_height, new_width))
        else:
            # Apply downsampling
            out = self.downsample(x)
        
        # Apply normalization
        out = self.norm(out)
        
        # Apply activation if specified
        if self.use_activation:
            out = self.activation(out)
            
        print(f"DownsamplingBlock output shape: {out.shape}")
        return out