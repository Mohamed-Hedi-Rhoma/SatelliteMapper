import torch
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# Import your classes
from Satellitetranslate.src.satellitetranslate.generatorconfig import GeneratorConfig
from Satellitetranslate.src.satellitetranslate.utils.sentinel_rsr import RSRReader
from Satellitetranslate.src.satellitetranslate.LandsatToSentinelGenerator import LandsatToSentinelGenerator


def test_landsat_to_sentinel_generator(
    landsat_tensor,
    landsat_angles,
    sentinel_angles,
    rsr_file_path,
    base_filters=64,
    num_res_blocks=9,
    use_hyperspectral=True,
    apply_rsr=True,
    use_attention=True,
    use_conditional_norm=True,
    separate_decoders=True
):
    """
    Test the Landsat to Sentinel generator with sample inputs.
    
    Args:
        landsat_tensor (torch.Tensor): Landsat image tensor [B, 6, 128, 128]
        landsat_angles (torch.Tensor): Landsat angular information [B, 4]
        sentinel_angles (torch.Tensor): Sentinel angular information [B, 4]
        rsr_file_path (str): Path to the RSR file
        base_filters (int): Base number of filters in the generator
        num_res_blocks (int): Number of residual blocks
        use_hyperspectral (bool): Whether to use hyperspectral representation
        apply_rsr (bool): Whether to apply RSR functions
        use_attention (bool): Whether to use attention mechanism
        use_conditional_norm (bool): Whether to use conditional normalization
        separate_decoders (bool): Whether to use separate decoders for VNIR and SWIR
        
    Returns:
        tuple: (sentinel_10m_img, sentinel_20m_img) - Generated Sentinel-2 bands at 10m and 20m resolution
    """
    # Check if the RSR file exists
    if not os.path.exists(rsr_file_path):
        raise FileNotFoundError(f"RSR file not found at: {rsr_file_path}")
    
    # Validate input tensors
    if landsat_tensor.ndim != 4 or landsat_tensor.shape[1] != 6:
        raise ValueError(f"Expected landsat_tensor with shape [B, 6, 128, 128], got {landsat_tensor.shape}")
    
    if landsat_angles.ndim != 2 or landsat_angles.shape[1] != 4:
        raise ValueError(f"Expected landsat_angles with shape [B, 4], got {landsat_angles.shape}")
    
    if sentinel_angles.ndim != 2 or sentinel_angles.shape[1] != 4:
        raise ValueError(f"Expected sentinel_angles with shape [B, 4], got {sentinel_angles.shape}")
    
    # Check batch size consistency
    batch_size = landsat_tensor.shape[0]
    if landsat_angles.shape[0] != batch_size or sentinel_angles.shape[0] != batch_size:
        raise ValueError("Batch size must be consistent across all input tensors")
    
    # Create RSR reader
    rsr_reader = RSRReader(rsr_file_path)
    
    # Print RSR information
    full_matrix, vnir_matrix, swir_matrix, wavelengths, vnir_wavelengths, swir_wavelengths = rsr_reader.read_rsr_file()
    print(f"RSR loaded successfully:")
    print(f"  VNIR matrix shape: {vnir_matrix.shape}")
    print(f"  SWIR matrix shape: {swir_matrix.shape}")
    print(f"  VNIR wavelength range: {vnir_wavelengths[0]:.2f} - {vnir_wavelengths[-1]:.2f} um")
    print(f"  SWIR wavelength range: {swir_wavelengths[0]:.2f} - {swir_wavelengths[-1]:.2f} um")
    
    # Create generator configuration
    config = GeneratorConfig(
        # Input/Output specifications
        landsat_channels=6,
        sentinel_vnir_channels=4,
        sentinel_swir_channels=2,
        angle_dim=4,
        
        # Hyperspectral representation
        vnir_hyperspectral_dim=vnir_matrix.shape[1] if apply_rsr else 100,
        swir_hyperspectral_dim=swir_matrix.shape[1] if apply_rsr else 50,
        use_hyperspectral=use_hyperspectral,
        
        # Resolution specifications
        landsat_resolution=128,
        sentinel_vnir_resolution=384,
        sentinel_swir_resolution=192,
        
        # Network architecture
        base_filters=base_filters,
        num_res_blocks=num_res_blocks,
        use_attention=use_attention,
        norm_type="instance",
        
        # Upsampling specifications
        vnir_upsampling_factor=3,
        vnir_upsample_mode="bilinear",
        swir_upsampling_factor=1.5,
        swir_upsample_mode="bilinear",
        
        # Conditional normalization
        use_conditional_norm=use_conditional_norm,
        cond_norm_position="all",
        
        # Physical constraints
        apply_rsr=apply_rsr,
        rsr_position="end",
        rsr_file_path=rsr_file_path,
        
        # Multi-resolution decoder options
        shared_encoder=True,
        shared_bottleneck=True,
        separate_decoders=separate_decoders
    )
    
    # Print configuration summary
    print(f"\nGenerator configuration:")
    print(f"  Hyperspectral: {config.use_hyperspectral}, Apply RSR: {config.apply_rsr}")
    print(f"  VNIR Hyperspectral dim: {config.vnir_hyperspectral_dim}")
    print(f"  SWIR Hyperspectral dim: {config.swir_hyperspectral_dim}")
    print(f"  Attention: {config.use_attention}, Conditional norm: {config.use_conditional_norm}")
    print(f"  Residual blocks: {config.num_res_blocks}, Base filters: {config.base_filters}")
    
    # Create the generator
    generator = LandsatToSentinelGenerator(config, rsr_reader)
    generator = generator.to(device)

    # Print model summary (parameter count)
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Set model to evaluation mode
    generator.eval()
    
    # Run the generator
    print("\nRunning generator...")
    with torch.no_grad():
        sentinel_10m_img, sentinel_20m_img = generator(landsat_tensor, sentinel_angles, landsat_angles)
    
    # Print output shapes
    print(f"  Input shape: {landsat_tensor.shape}")
    print(f"  Sentinel 10m output shape: {sentinel_10m_img.shape}")
    print(f"  Sentinel 20m output shape: {sentinel_20m_img.shape}")
    print(f"  Expected 10m shape: [B, 4, 384, 384]")
    print(f"  Expected 20m shape: [B, 2, 192, 192]")
    
    # Verify output ranges
    print(f"\nOutput value ranges:")
    print(f"  Sentinel 10m min/max: {sentinel_10m_img.min().item():.3f}/{sentinel_10m_img.max().item():.3f}")
    print(f"  Sentinel 20m min/max: {sentinel_20m_img.min().item():.3f}/{sentinel_20m_img.max().item():.3f}")
    print(f"  Expected range: -1.0/1.0 (from tanh activation)")
    
    return sentinel_10m_img, sentinel_20m_img


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy input tensors
    batch_size = 2
    landsat_img = torch.randn(batch_size, 6, 128, 128, device=device)  # [B, 6, 128, 128]
    landsat_angles = torch.randn(batch_size, 4, device=device)          # [B, 4]
    sentinel_angles = torch.randn(batch_size, 4, device=device)         # [B, 4]
    
    # Path to RSR file
    rsr_file_path = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/pth_data/sentinel2.rsr"
    
    # Test the generator
    sentinel_10m, sentinel_20m = test_landsat_to_sentinel_generator(
        landsat_img,
        landsat_angles,
        sentinel_angles,
        rsr_file_path,
        base_filters=64,
        num_res_blocks=9,
        use_hyperspectral=True,
        apply_rsr=True,
        use_attention=True,
        use_conditional_norm=True,
        separate_decoders=True
    )
    
    print("\nGenerator test completed successfully!")