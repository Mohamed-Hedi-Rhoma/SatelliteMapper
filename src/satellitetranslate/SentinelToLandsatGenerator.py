import torch
import torch.nn as nn
import torch.nn.functional as F

from ConditionalInstanceNorm2d import ConditionalInstanceNorm2d
from ResidualBlock import ResidualBlock
from AttentionBlock import AttentionBlock
from SpectralResponseLayer import SpectralResponseLayer
from DownsamplingBlock import DownsamplingBlock
from utils.landsat8_rsr import Landsat8RSRReader


class SentinelToLandsatGenerator(nn.Module):
    """
    Physics-informed generator for translating Sentinel-2 images to Landsat-8.
    
    This generator:
    1. Takes Sentinel-2 imagery (at two resolutions) and angular information as input
    2. Aligns the multi-resolution inputs to a common feature space
    3. Processes through residual blocks and attention mechanisms
    4. Creates a hyperspectral representation
    5. Applies Landsat-8 RSR functions to generate Landsat-8 bands
    
    Args:
        config: GeneratorConfig object containing architecture parameters
        rsr_reader: RSRReader object for loading Landsat-8 spectral response functions
    """
    def __init__(self, config, rsr_reader=None):
        super(SentinelToLandsatGenerator, self).__init__()
        self.config = config
        
        # Create RSR layer if provided with reader
        if rsr_reader and config.apply_rsr:
            landsat_matrix, wavelengths = rsr_reader.read_rsr_file()
            
            # Set hyperspectral dimension based on RSR matrix
            self.config.landsat_hyperspectral_dim = landsat_matrix.shape[1]
            
            # Create RSR layer
            self.landsat_rsr_layer = SpectralResponseLayer(
                torch.from_numpy(landsat_matrix).float(),
                trainable=False
            )
            
            print(f"Loaded Landsat-8 RSR matrix with shape: {landsat_matrix.shape}")
            print(f"Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} micrometers")
        elif config.use_hyperspectral:
            # If no RSR reader but still using hyperspectral approach,
            # ensure dimension is set to a reasonable value
            if self.config.landsat_hyperspectral_dim == 0:
                self.config.landsat_hyperspectral_dim = 150
        
        # Initial convolutional layers for VNIR (10m) and SWIR (20m) bands
        self.vnir_initial_conv = nn.Conv2d(
            config.sentinel_vnir_channels, 
            config.base_filters // 2,  # Half the filters for each branch
            kernel_size=7, 
            padding=3
        )
        
        self.swir_initial_conv = nn.Conv2d(
            config.sentinel_swir_channels, 
            config.base_filters // 2,  # Half the filters for each branch
            kernel_size=7, 
            padding=3
        )
        
        # Normalization for initial convolutions
        if config.norm_type == 'instance':
            self.vnir_initial_norm = nn.InstanceNorm2d(config.base_filters // 2)
            self.swir_initial_norm = nn.InstanceNorm2d(config.base_filters // 2)
        elif config.norm_type == 'batch':
            self.vnir_initial_norm = nn.BatchNorm2d(config.base_filters // 2)
            self.swir_initial_norm = nn.BatchNorm2d(config.base_filters // 2)
        else:
            self.vnir_initial_norm = nn.Identity()
            self.swir_initial_norm = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
        # Downsampling from Sentinel-2 resolutions to Landsat-8 resolution
        self.vnir_downsample = DownsamplingBlock(
            in_channels=config.base_filters // 2,
            out_channels=config.base_filters // 2,
            scale_factor=3.0,  # 10m to 30m
            mode='conv',
            norm_type=config.norm_type
        )
        
        self.swir_downsample = DownsamplingBlock(
            in_channels=config.base_filters // 2,
            out_channels=config.base_filters // 2,
            scale_factor=1.5,  # 20m to 30m
            mode='adaptive',  # For non-integer scaling
            norm_type=config.norm_type
        )
        
        # Fusion layer to combine VNIR and SWIR features
        self.fusion_conv = nn.Conv2d(
            config.base_filters, 
            config.base_filters,
            kernel_size=3, 
            padding=1
        )
        
        if config.norm_type == 'instance':
            self.fusion_norm = nn.InstanceNorm2d(config.base_filters)
        elif config.norm_type == 'batch':
            self.fusion_norm = nn.BatchNorm2d(config.base_filters)
        else:
            self.fusion_norm = nn.Identity()
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(config.num_res_blocks):
            self.residual_blocks.append(
                ResidualBlock(
                    channels=config.base_filters,
                    angle_dim=config.angle_dim,
                    use_cond_norm=config.use_conditional_norm,
                    norm_type=config.norm_type
                )
            )
        
        # Attention block (optional)
        if config.use_attention:
            self.attention = AttentionBlock(config.base_filters)
        
        # Hyperspectral projection
        if config.use_hyperspectral:
            self.hyperspectral_proj = nn.Conv2d(
                config.base_filters,
                self.config.landsat_hyperspectral_dim,
                kernel_size=1
            )
        
        # Final convolution (if not using RSR)
        if not config.apply_rsr:
            self.final_conv = nn.Conv2d(
                self.config.landsat_hyperspectral_dim if config.use_hyperspectral else config.base_filters,
                config.landsat_channels,
                kernel_size=7,
                padding=3
            )
    
    def forward(self, sentinel_10m_img, sentinel_20m_img, landsat_angles=None, sentinel_angles=None):
        """
        Forward pass of the generator.
        
        Args:
            sentinel_10m_img: Sentinel-2 10m resolution bands [B, vnir_channels, H*3, W*3]
            sentinel_20m_img: Sentinel-2 20m resolution bands [B, swir_channels, H*1.5, W*1.5]
            landsat_angles: Landsat-8 angle tensor [B, angle_dim]
            sentinel_angles: Sentinel-2 angle tensor [B, angle_dim]
            
        Returns:
            torch.Tensor: Landsat-8 image tensor [B, landsat_channels, H, W]
        """
        # Use the target domain angles (Landsat-8) for conditional normalization
        angles = landsat_angles
        
        # Print input shapes for debugging
        print(f"Sentinel 10m input shape: {sentinel_10m_img.shape}")
        print(f"Sentinel 20m input shape: {sentinel_20m_img.shape}")
        
        # Process VNIR branch (10m)
        vnir_x = self.vnir_initial_conv(sentinel_10m_img)
        vnir_x = self.vnir_initial_norm(vnir_x)
        vnir_x = self.relu(vnir_x)
        print(f"After VNIR initial conv: {vnir_x.shape}")
        
        # Process SWIR branch (20m)
        swir_x = self.swir_initial_conv(sentinel_20m_img)
        swir_x = self.swir_initial_norm(swir_x)
        swir_x = self.relu(swir_x)
        print(f"After SWIR initial conv: {swir_x.shape}")
        
        # Downsample to Landsat-8 resolution
        vnir_x = self.vnir_downsample(vnir_x)
        print(f"After VNIR downsampling: {vnir_x.shape}")
        
        swir_x = self.swir_downsample(swir_x)
        print(f"After SWIR downsampling: {swir_x.shape}")
        
        # Concatenate VNIR and SWIR features along channel dimension
        x = torch.cat([vnir_x, swir_x], dim=1)
        print(f"After concatenation: {x.shape}")
        
        # Apply fusion convolution
        x = self.fusion_conv(x)
        x = self.fusion_norm(x)
        x = self.relu(x)
        print(f"After fusion: {x.shape}")
        
        # Residual blocks
        for i, res_block in enumerate(self.residual_blocks):
            if self.config.use_conditional_norm and angles is not None:
                x = res_block(x, angles)
            else:
                x = res_block(x)
            if i == 0 or i == len(self.residual_blocks) - 1:
                print(f"After residual block {i}: {x.shape}")
        
        # Attention (optional)
        if self.config.use_attention:
            x = self.attention(x)
            print(f"After attention: {x.shape}")
        
        # Hyperspectral projection
        if self.config.use_hyperspectral:
            x = self.hyperspectral_proj(x)
            print(f"After hyperspectral projection: {x.shape}")
        
        # Apply RSR or final convolution
        if self.config.apply_rsr and hasattr(self, 'landsat_rsr_layer'):
            landsat_img = self.landsat_rsr_layer(x)
            print(f"After RSR application: {landsat_img.shape}")
        else:
            # If not using RSR, apply final convolution
            if hasattr(self, 'final_conv'):
                landsat_img = self.final_conv(x)
                print(f"After final convolution: {landsat_img.shape}")
            else:
                # If we reach here, it means configuration is inconsistent
                raise ValueError("Invalid configuration: either apply_rsr=True with RSR layer or "
                               "apply_rsr=False with final convolution layer required")
        
        # Apply tanh to get values in [-1, 1] range (assuming input is normalized to this range)
        landsat_img = torch.tanh(landsat_img)
        
        return landsat_img


# Example usage:
if __name__ == "__main__":
    from satellitetranslate.generatorconfig import GeneratorConfig
    
    # Create a configuration
    config = GeneratorConfig(
        landsat_channels=6,
        sentinel_vnir_channels=4,
        sentinel_swir_channels=2,
        landsat_hyperspectral_dim=100,
        use_hyperspectral=True,
        apply_rsr=False,  # Not using RSR for this example
    )
    
    # Create the generator
    generator = SentinelToLandsatGenerator(config)
    
    # Create dummy input tensors
    sentinel_10m_img = torch.randn(2, 4, 384, 384)  # [batch_size, channels, height, width]
    sentinel_20m_img = torch.randn(2, 2, 192, 192)  # [batch_size, channels, height, width]
    landsat_angles = torch.randn(2, 4)              # [batch_size, angle_dim]
    
    # Forward pass
    landsat_img = generator(sentinel_10m_img, sentinel_20m_img, landsat_angles)
    
    # Print shapes
    print(f"\nSentinel 10m input shape: {sentinel_10m_img.shape}")
    print(f"Sentinel 20m input shape: {sentinel_20m_img.shape}")
    print(f"Landsat output shape: {landsat_img.shape}")
    print(f"Expected Landsat shape: [2, 6, 128, 128]")