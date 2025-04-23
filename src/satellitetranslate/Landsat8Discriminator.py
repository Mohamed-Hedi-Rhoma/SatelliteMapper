import torch
import torch.nn as nn
import torch.nn.functional as F
from ResidualDiscriminatorBlock import ResidualDiscriminatorBlock

class Landsat8Discriminator(nn.Module):
    """
    PatchGAN discriminator for Landsat-8 imagery with residual connections.
    """
    def __init__(self, config):
        super(Landsat8Discriminator, self).__init__()
        self.config = config
        
        # For Landsat-8, all bands have the same resolution
        self.input_channels = config.landsat_channels  # 6 bands for Landsat-8
        self.input_resolution = config.landsat_resolution  # 128×128
        print(f"Creating Landsat-8 discriminator for {self.input_resolution}×{self.input_resolution} input with {self.input_channels} channels")
        
        # Initial convolution to get to base_filters
        self.initial_conv = nn.Sequential(
            nn.Conv2d(self.input_channels, config.base_filters, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Calculate number of downsampling layers based on input resolution
        target_resolution = 24  # Target resolution for final feature maps
        self.n_layers = min(
            config.n_layers,
            max(1, int(torch.log2(torch.tensor(self.input_resolution / target_resolution)).item()))
        )
        print(f"Using {self.n_layers} downsampling layers for Landsat-8 discriminator")
        
        # Build the downsampling layers with increasing channel counts
        self.down_blocks = nn.ModuleList()
        in_channels = config.base_filters
        
        for i in range(self.n_layers):
            # Double the number of filters each time, up to max_filters
            out_channels = min(in_channels * 2, config.max_filters)
            
            # Print layer information
            print(f"Layer {i+1}: {in_channels} → {out_channels} channels")
            
            # Downsampling convolutional layer
            self.down_blocks.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
            )
            
            # Add residual blocks at each resolution level
            res_blocks = nn.ModuleList()
            for j in range(2):  # Add 2 residual blocks at each level
                res_blocks.append(
                    ResidualDiscriminatorBlock(
                        channels=out_channels,
                        stride=1,  # No downsampling in residual blocks
                        angle_dim=config.angle_dim,
                        use_cond_norm=config.use_conditional_disc,
                        norm_type=config.norm_type,
                        block_name=f"landsat_layer{i+1}_res{j+1}"  # Add meaningful block names
                    )
                )
            self.down_blocks.append(res_blocks)
            
            in_channels = out_channels
        
        # Final layers
        self.final_conv = nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)
        self.use_sigmoid = config.use_sigmoid
    
    def forward(self, x, angles=None):
        """Forward pass of the discriminator."""
        # Print input shape
        print(f"\nLandsat-8 Discriminator forward pass:")
        print(f"Input shape: {x.shape}")
        
        # Initial convolution
        x = self.initial_conv(x)
        print(f"After initial conv: {x.shape}")
        
        # Downsampling with residual blocks
        for i in range(0, len(self.down_blocks), 2):
            # Downsampling conv
            downsample = self.down_blocks[i]
            x = downsample(x)
            x = F.leaky_relu(x, 0.2, inplace=True)
            print(f"After downsampling {i//2 + 1}: {x.shape}")
            
            # Residual blocks
            res_blocks = self.down_blocks[i+1]
            for j, res_block in enumerate(res_blocks):
                x = res_block(x, angles)
                print(f"After residual block {i//2 + 1}-{j+1}: {x.shape}")
        
        # Final convolution
        x = self.final_conv(x)
        print(f"After final conv: {x.shape}")
        
        # Apply sigmoid if specified
        if self.use_sigmoid:
            x = torch.sigmoid(x)
            print(f"After sigmoid: {x.shape}")
        
        return x