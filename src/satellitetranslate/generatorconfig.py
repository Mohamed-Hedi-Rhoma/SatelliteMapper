from dataclasses import dataclass
from typing import List, Tuple, Optional, Union


@dataclass
class GeneratorConfig:
    """Configuration for the Landsat to Sentinel-2 Generator with multi-resolution outputs."""
    # Input/Output specifications
    landsat_channels: int = 6              # Number of input Landsat-8 bands
    sentinel_vnir_channels: int = 4        # Number of visible/NIR Sentinel-2 bands (blue, green, red, NIR)
    sentinel_swir_channels: int = 2        # Number of SWIR Sentinel-2 bands (SWIR1, SWIR2)
    angle_dim: int = 4                     # Dimension of angular information
    
    # Hyperspectral representation
    vnir_hyperspectral_dim: int = 0        # Dimension of VNIR hyperspectral representation (auto-set from RSR)
    swir_hyperspectral_dim: int = 0        # Dimension of SWIR hyperspectral representation (auto-set from RSR)
    use_hyperspectral: bool = True         # Whether to use hyperspectral representation
    
    # Resolution specifications
    landsat_resolution: int = 128          # Landsat-8 resolution
    sentinel_vnir_resolution: int = 384    # Sentinel-2 VNIR resolution (384×384)
    sentinel_swir_resolution: int = 192    # Sentinel-2 SWIR resolution (192×192)
    
    # Network architecture
    base_filters: int = 64                 # Base number of filters in conv layers
    num_res_blocks: int = 9                # Number of residual blocks
    use_attention: bool = True             # Whether to use attention mechanism
    norm_type: str = "instance"            # Normalization type: "batch", "instance", or "none"
    
    # Upsampling specifications for VNIR branch
    vnir_upsampling_factor: int = 3        # Landsat→Sentinel VNIR resolution ratio (384/128)
    vnir_upsample_mode: str = "bilinear"   # Upsampling mode for VNIR: "nearest", "bilinear", "transpose"
    
    # Upsampling specifications for SWIR branch
    swir_upsampling_factor: int = 1.5      # Landsat→Sentinel SWIR resolution ratio (192/128)
    swir_upsample_mode: str = "bilinear"   # Upsampling mode for SWIR: "nearest", "bilinear", "transpose"
    
    # Conditional normalization
    use_conditional_norm: bool = True      # Whether to use conditional normalization
    cond_norm_position: str = "all"        # Where to apply cond norm: "input", "middle", "all"
    
    # Physical constraints
    apply_rsr: bool = True                 # Apply Relative Spectral Response functions
    rsr_position: str = "end"              # Where to apply RSR: "middle" or "end"
    rsr_file_path: str = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/pth_data/sentinel2.rsr"   # Path to the RSR file
    
    # Multi-resolution decoder options
    shared_encoder: bool = True            # Whether to use a shared encoder for both resolutions
    shared_bottleneck: bool = True         # Whether to use a shared bottleneck/hyperspectral layer
    separate_decoders: bool = True         # Whether to use separate decoders for VNIR and SWIR