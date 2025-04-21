import torch
import torch.nn as nn
import torch.nn.functional as F

from Satellitetranslate.src.satellitetranslate.ConditionalInstanceNorm2d import ConditionalInstanceNorm2d
from Satellitetranslate.src.satellitetranslate.ResidualBlock import ResidualBlock
from Satellitetranslate.src.satellitetranslate.AttentionBlock import AttentionBlock
from Satellitetranslate.src.satellitetranslate.SpectralResponseLayer import SpectralResponseLayer
from Satellitetranslate.src.satellitetranslate.UpsamplingBlock import UpsamplingBlock, MultiscaleUpsamplingBlock


class LandsatToSentinelGenerator(nn.Module):
    """
    Physics-informed generator for translating Landsat-8 images to Sentinel-2.
    
    This generator:
    1. Takes Landsat-8 imagery and angular information as input
    2. Processes through encoder, residual blocks, and attention mechanisms
    3. Creates a hyperspectral representation
    4. Applies RSR functions to generate Sentinel-2 bands at appropriate resolutions
    
    Args:
        config: GeneratorConfig object containing architecture parameters
        rsr_reader: RSRReader object for loading Sentinel-2 spectral response functions
    """
    def __init__(self, config, rsr_reader=None):
        super(LandsatToSentinelGenerator, self).__init__()
        self.config = config
        
        # Create RSR layers if provided with reader
        if rsr_reader and config.apply_rsr:
            _, vnir_matrix, swir_matrix, _, _, _ = rsr_reader.read_rsr_file()
            
            # Set hyperspectral dimensions based on RSR matrices
            self.config.vnir_hyperspectral_dim = vnir_matrix.shape[1]
            self.config.swir_hyperspectral_dim = swir_matrix.shape[1]
            
            # Create RSR layers
            self.vnir_rsr_layer = SpectralResponseLayer(
                torch.from_numpy(vnir_matrix).float(),
                trainable=False
            )
            
            self.swir_rsr_layer = SpectralResponseLayer(
                torch.from_numpy(swir_matrix).float(),
                trainable=False
            )
        elif config.use_hyperspectral:
            # If no RSR reader but still using hyperspectral approach,
            # ensure dimensions are set to reasonable values
            if self.config.vnir_hyperspectral_dim == 0:
                self.config.vnir_hyperspectral_dim = 100
            if self.config.swir_hyperspectral_dim == 0:
                self.config.swir_hyperspectral_dim = 50
        
        # Initial convolutional layer
        self.initial_conv = nn.Conv2d(
            config.landsat_channels, 
            config.base_filters,
            kernel_size=7, 
            padding=3
        )
        
        if config.norm_type == 'instance':
            self.initial_norm = nn.InstanceNorm2d(config.base_filters)
        elif config.norm_type == 'batch':
            self.initial_norm = nn.BatchNorm2d(config.base_filters)
        else:
            self.initial_norm = nn.Identity()
            
        self.relu = nn.ReLU(inplace=True)
        
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
        
        # Attention blocks (optional)
        if config.use_attention:
            self.attention = AttentionBlock(config.base_filters)
        
        # Hyperspectral projection
        if config.use_hyperspectral:
            # For separate decoders, create two projection layers
            if config.separate_decoders:
                self.vnir_hyperspectral_proj = nn.Conv2d(
                    config.base_filters,
                    self.config.vnir_hyperspectral_dim,
                    kernel_size=1
                )
                
                self.swir_hyperspectral_proj = nn.Conv2d(
                    config.base_filters,
                    self.config.swir_hyperspectral_dim,
                    kernel_size=1
                )
            else:
                # Use a single, shared hyperspectral projection
                # We'll take the max of VNIR and SWIR dimensions
                shared_hyperspectral_dim = max(
                    self.config.vnir_hyperspectral_dim,
                    self.config.swir_hyperspectral_dim
                )
                
                self.hyperspectral_proj = nn.Conv2d(
                    config.base_filters,
                    shared_hyperspectral_dim,
                    kernel_size=1
                )
        
        # Upsampling blocks
        if config.separate_decoders:
            # Use multi-scale upsampling with two separate branches
            self.multiscale_upsample = MultiscaleUpsamplingBlock(
                in_channels=config.base_filters if not config.use_hyperspectral else max(
                    self.config.vnir_hyperspectral_dim,
                    self.config.swir_hyperspectral_dim
                ),
                vnir_channels=self.config.vnir_hyperspectral_dim if config.use_hyperspectral else config.sentinel_vnir_channels,
                swir_channels=self.config.swir_hyperspectral_dim if config.use_hyperspectral else config.sentinel_swir_channels,
                vnir_scale=config.vnir_upsampling_factor,
                swir_scale=config.swir_upsampling_factor,
                mode=config.vnir_upsample_mode,
                norm_type=config.norm_type
            )
        else:
            # Use separate upsampling blocks for each branch
            self.vnir_upsample = UpsamplingBlock(
                in_channels=config.base_filters if not config.use_hyperspectral else self.config.vnir_hyperspectral_dim,
                out_channels=self.config.vnir_hyperspectral_dim if config.use_hyperspectral else config.sentinel_vnir_channels,
                scale_factor=config.vnir_upsampling_factor,
                mode=config.vnir_upsample_mode,
                norm_type=config.norm_type
            )
            
            self.swir_upsample = UpsamplingBlock(
                in_channels=config.base_filters if not config.use_hyperspectral else self.config.swir_hyperspectral_dim,
                out_channels=self.config.swir_hyperspectral_dim if config.use_hyperspectral else config.sentinel_swir_channels,
                scale_factor=config.swir_upsampling_factor,
                mode=config.swir_upsample_mode,
                norm_type=config.norm_type
            )
        
        # Final convolution layers (if not using RSR)
        if not config.apply_rsr:
            self.final_vnir_conv = nn.Conv2d(
                self.config.vnir_hyperspectral_dim if config.use_hyperspectral else config.base_filters,
                config.sentinel_vnir_channels,
                kernel_size=7,
                padding=3
            )
            
            self.final_swir_conv = nn.Conv2d(
                self.config.swir_hyperspectral_dim if config.use_hyperspectral else config.base_filters,
                config.sentinel_swir_channels,
                kernel_size=7,
                padding=3
            )
    
    def forward(self, landsat_img, sentinel_angles=None, landsat_angles=None):
        """
        Forward pass of the generator.
        
        Args:
            landsat_img: Landsat-8 image tensor of shape [B, landsat_channels, H, W]
            sentinel_angles: Sentinel-2 angle tensor of shape [B, angle_dim]
            landsat_angles: Landsat-8 angle tensor of shape [B, angle_dim]
            
        Returns:
            tuple: Two tensors:
                - sentinel_10m_img: Sentinel-2 10m resolution bands [B, vnir_channels, H*3, W*3]
                - sentinel_20m_img: Sentinel-2 20m resolution bands [B, swir_channels, H*1.5, W*1.5]
        """
        # Use the target domain angles (Sentinel-2) for conditional normalization
        angles = sentinel_angles
        
        # Initial convolution
        x = self.initial_conv(landsat_img)
        print(f"After initial_conv: {x.shape}")
        x = self.initial_norm(x)
        x = self.relu(x)
        
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
            if self.config.separate_decoders:
                # Create separate hyperspectral representations for VNIR and SWIR
                vnir_hyper = self.vnir_hyperspectral_proj(x)
                swir_hyper = self.swir_hyperspectral_proj(x)
                print(f"vnir_hyper shape: {vnir_hyper.shape}")
                print(f"swir_hyper shape: {swir_hyper.shape}")
                
                # Upsampling (using hyperspectral features - THIS WAS THE BUG)
                if hasattr(self, 'multiscale_upsample'):
                    # FIX: Use max dimension hyperspectral for multiscale
                    if self.config.vnir_hyperspectral_dim >= self.config.swir_hyperspectral_dim:
                        print(f"Using vnir_hyper for multiscale_upsample: {vnir_hyper.shape}")
                        vnir_features, swir_features = self.multiscale_upsample(vnir_hyper)
                    else:
                        print(f"Using swir_hyper for multiscale_upsample: {swir_hyper.shape}")
                        vnir_features, swir_features = self.multiscale_upsample(swir_hyper)
                else:
                    vnir_features = self.vnir_upsample(vnir_hyper)
                    swir_features = self.swir_upsample(swir_hyper)
            else:
                # Create a shared hyperspectral representation
                hyper = self.hyperspectral_proj(x)
                print(f"hyper shape: {hyper.shape}")
                
                # Upsampling
                if hasattr(self, 'multiscale_upsample'):
                    vnir_features, swir_features = self.multiscale_upsample(hyper)
                else:
                    vnir_features = self.vnir_upsample(hyper)
                    swir_features = self.swir_upsample(hyper)
        else:
            # No hyperspectral representation, just use the features directly
            print(f"No hyperspectral, using x directly: {x.shape}")
            if hasattr(self, 'multiscale_upsample'):
                vnir_features, swir_features = self.multiscale_upsample(x)
            else:
                vnir_features = self.vnir_upsample(x)
                swir_features = self.swir_upsample(x)
        
        print(f"vnir_features shape: {vnir_features.shape}")
        print(f"swir_features shape: {swir_features.shape}")
        
        # Apply RSR functions or final convolution
        if self.config.apply_rsr and hasattr(self, 'vnir_rsr_layer') and hasattr(self, 'swir_rsr_layer'):
            sentinel_10m_img = self.vnir_rsr_layer(vnir_features)
            sentinel_20m_img = self.swir_rsr_layer(swir_features)
            print(f"After RSR - sentinel_10m_img: {sentinel_10m_img.shape}")
            print(f"After RSR - sentinel_20m_img: {sentinel_20m_img.shape}")
        else:
            # If not using RSR, apply final convolution
            if hasattr(self, 'final_vnir_conv') and hasattr(self, 'final_swir_conv'):
                sentinel_10m_img = self.final_vnir_conv(vnir_features)
                sentinel_20m_img = self.final_swir_conv(swir_features)
                print(f"After final conv - sentinel_10m_img: {sentinel_10m_img.shape}")
                print(f"After final conv - sentinel_20m_img: {sentinel_20m_img.shape}")
            else:
                # If we reach here, it means configuration is inconsistent
                raise ValueError("Invalid configuration: either apply_rsr=True with RSR layers or "
                            "apply_rsr=False with final convolution layers required")
        
        # Apply tanh to get values in [-1, 1] range (assuming input is normalized to this range)
        sentinel_10m_img = torch.tanh(sentinel_10m_img)
        sentinel_20m_img = torch.tanh(sentinel_20m_img)
        
        return sentinel_10m_img, sentinel_20m_img


# Example usage:
if __name__ == "__main__":
    from Satellitetranslate.src.satellitetranslate.generatorconfig import GeneratorConfig
    
    # Create a configuration
    config = GeneratorConfig(
        landsat_channels=6,
        sentinel_vnir_channels=4,
        sentinel_swir_channels=2,
        vnir_hyperspectral_dim=100,
        swir_hyperspectral_dim=50,
        use_hyperspectral=True,
        apply_rsr=False,  # Not using RSR for this example
        separate_decoders=True
    )
    
    # Create the generator
    generator = LandsatToSentinelGenerator(config)
    
    # Create dummy input tensors
    landsat_img = torch.randn(2, 6, 128, 128)  # [batch_size, channels, height, width]
    sentinel_angles = torch.randn(2, 4)         # [batch_size, angle_dim]
    
    # Forward pass
    sentinel_10m_img, sentinel_20m_img = generator(landsat_img, sentinel_angles)
    
    # Print shapes
    print(f"Input shape: {landsat_img.shape}")
    print(f"Sentinel 10m output shape: {sentinel_10m_img.shape}")
    print(f"Sentinel 20m output shape: {sentinel_20m_img.shape}")
    print(f"Expected 10m shape: [2, 4, 384, 384]")
    print(f"Expected 20m shape: [2, 2, 192, 192]")