from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

@dataclass
class DiscriminatorConfig:
    """Configuration for satellite image discriminators with multi-resolution support."""
    
    # Input specifications for Sentinel-2
    vnir_channels: int = 4             # Number of VNIR channels (Blue, Green, Red, NIR)
    swir_channels: int = 2             # Number of SWIR channels (SWIR1, SWIR2)
    
    # Input specifications for Landsat-8
    landsat_channels: int = 6          # Number of Landsat-8 bands
    
    # Angular information
    angle_dim: int = 4                 # Dimension of angular information
    
    # Resolution specifications
    vnir_resolution: int = 384         # VNIR resolution (384×384)
    swir_resolution: int = 192         # SWIR resolution (192×192)
    landsat_resolution: int = 128      # Landsat-8 resolution (128×128)
    
    # Network architecture
    base_filters: int = 64             # Base number of filters in conv layers
    max_filters: int = 512             # Maximum number of filters in any layer
    n_layers: int = 4                  # Number of downsampling layers
    kernel_size: int = 4               # Kernel size for convolutions
    
    # Discrimination type
    use_patch_gan: bool = True         # Whether to use PatchGAN (multiple outputs)
    
    # Conditional discrimination
    use_conditional_disc: bool = True  # Whether to use conditional discrimination
    
    # Normalization and activation
    norm_type: str = "instance"        # Normalization type: "batch", "instance", or "none"
    use_sigmoid: bool = False          # Whether to use sigmoid at the output (False for LSGAN)
    
    # Multi-resolution discriminator options
    shared_params: bool = False        # Whether different discriminators share parameters