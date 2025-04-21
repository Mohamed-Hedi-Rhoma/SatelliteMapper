import torch
import torch.nn as nn
import torch.nn.functional as F


class UpsamplingBlock(nn.Module):
    """
    Upsampling block for increasing spatial resolution.
    
    This block increases the spatial dimensions of feature maps while maintaining
    or changing the number of channels. It supports different upsampling methods.
    """
    def __init__(
        self, 
        in_channels, 
        out_channels=None, 
        scale_factor=2.0, 
        mode='bilinear', 
        norm_type='instance', 
        use_activation=True
    ):
        super(UpsamplingBlock, self).__init__()
        
        # If out_channels is not specified, use the same as in_channels
        out_channels = out_channels or in_channels
        
        # Print initialization info for debugging
        print(f"Creating UpsamplingBlock: in_channels={in_channels}, out_channels={out_channels}, scale={scale_factor}")
        
        # Create the upsampling layers
        if mode == 'transpose':
            # For transpose convolution, we need to determine the stride
            stride = round(scale_factor)
            print(f"Using transpose convolution with stride={stride}")
            self.upsample = nn.ConvTranspose2d(
                in_channels, 
                out_channels, 
                kernel_size=4,
                stride=stride,
                padding=1
            )
        else:
            # For nearest or bilinear upsampling
            print(f"Using {mode} upsampling with Conv2d")
            self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=scale_factor, 
                    mode=mode, 
                    align_corners=True if mode == 'bilinear' else None
                ),
                # Add a convolution to change the number of channels and refine features
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    padding=1
                )
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
        if use_activation:
            self.activation = nn.ReLU(inplace=True)
            
    def forward(self, x):
        """
        Forward pass of the upsampling block.
        """
        # Print input shape for debugging
        print(f"UpsamplingBlock input shape: {x.shape}")
        
        # Apply upsampling
        out = self.upsample(x)
        print(f"After upsampling shape: {out.shape}")
        
        # Apply normalization
        out = self.norm(out)
        
        # Apply activation if specified
        if self.use_activation:
            out = self.activation(out)
            
        return out


class MultiscaleUpsamplingBlock(nn.Module):
    """
    Specialized upsampling block for the different resolutions in Sentinel-2.
    
    This block handles the upsampling from Landsat resolution to the different
    Sentinel-2 resolutions (10m VNIR bands and 20m SWIR bands).
    """
    def __init__(
        self,
        in_channels,
        vnir_channels,
        swir_channels,
        vnir_scale=3.0,
        swir_scale=1.5,
        mode='bilinear',
        norm_type='instance'
    ):
        super(MultiscaleUpsamplingBlock, self).__init__()
        
        # Print initialization parameters for debugging
        print(f"\nCreating MultiscaleUpsamplingBlock:")
        print(f"  in_channels={in_channels}")
        print(f"  vnir_channels={vnir_channels}")
        print(f"  swir_channels={swir_channels}")
        print(f"  vnir_scale={vnir_scale}")
        print(f"  swir_scale={swir_scale}")
        print(f"  mode={mode}")
        
        # VNIR branch (higher resolution)
        self.vnir_upsample = UpsamplingBlock(
            in_channels=in_channels,
            out_channels=vnir_channels,
            scale_factor=vnir_scale,
            mode=mode,
            norm_type=norm_type
        )
        
        # SWIR branch (lower resolution)
        self.swir_upsample = UpsamplingBlock(
            in_channels=in_channels,
            out_channels=swir_channels,
            scale_factor=swir_scale,
            mode=mode,
            norm_type=norm_type
        )
    
    def forward(self, x):
        """
        Forward pass of the multiscale upsampling block.
        """
        # Print input shape for debugging
        print(f"\nMultiscaleUpsamplingBlock input shape: {x.shape}")
        
        print("Processing VNIR branch:")
        vnir_features = self.vnir_upsample(x)
        print(f"VNIR features output shape: {vnir_features.shape}")
        
        print("Processing SWIR branch:")
        swir_features = self.swir_upsample(x)
        print(f"SWIR features output shape: {swir_features.shape}")
        
        return vnir_features, swir_features