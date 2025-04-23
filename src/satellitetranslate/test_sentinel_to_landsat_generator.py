import torch
import os
from dataclasses import dataclass
import sys

# Import your classes
from generatorconfig import GeneratorConfig
from utils.landsat8_rsr import Landsat8RSRReader
from SentinelToLandsatGenerator import SentinelToLandsatGenerator
from DownsamplingBlock import DownsamplingBlock


def test_sentinel_to_landsat_generator(
    sentinel_10m_tensor,
    sentinel_20m_tensor,
    landsat_angles,
    sentinel_angles,
    rsr_file_path,
    base_filters=64,
    num_res_blocks=9,
    use_hyperspectral=True,
    apply_rsr=True,
    use_attention=True,
    use_conditional_norm=True
):
    """
    Test the Sentinel to Landsat generator with sample inputs.
    
    Args:
        sentinel_10m_tensor (torch.Tensor): Sentinel-2 10m bands tensor [B, 4, 384, 384]
        sentinel_20m_tensor (torch.Tensor): Sentinel-2 20m bands tensor [B, 2, 192, 192]
        landsat_angles (torch.Tensor): Landsat angular information [B, 4]
        sentinel_angles (torch.Tensor): Sentinel angular information [B, 4]
        rsr_file_path (str): Path to the Landsat-8 RSR file
        base_filters (int): Base number of filters in the generator
        num_res_blocks (int): Number of residual blocks
        use_hyperspectral (bool): Whether to use hyperspectral representation
        apply_rsr (bool): Whether to apply RSR functions
        use_attention (bool): Whether to use attention mechanism
        use_conditional_norm (bool): Whether to use conditional normalization
        
    Returns:
        torch.Tensor: landsat_img - Generated Landsat-8 image
    """
    # Check if the RSR file exists
    if not os.path.exists(rsr_file_path):
        raise FileNotFoundError(f"RSR file not found at: {rsr_file_path}")
    
    # Validate input tensors
    if sentinel_10m_tensor.ndim != 4 or sentinel_10m_tensor.shape[1] != 4:
        raise ValueError(f"Expected sentinel_10m_tensor with shape [B, 4, 384, 384], got {sentinel_10m_tensor.shape}")
    
    if sentinel_20m_tensor.ndim != 4 or sentinel_20m_tensor.shape[1] != 2:
        raise ValueError(f"Expected sentinel_20m_tensor with shape [B, 2, 192, 192], got {sentinel_20m_tensor.shape}")
    
    if landsat_angles.ndim != 2 or landsat_angles.shape[1] != 4:
        raise ValueError(f"Expected landsat_angles with shape [B, 4], got {landsat_angles.shape}")
    
    if sentinel_angles.ndim != 2 or sentinel_angles.shape[1] != 4:
        raise ValueError(f"Expected sentinel_angles with shape [B, 4], got {sentinel_angles.shape}")
    
    # Check batch size consistency
    batch_size = sentinel_10m_tensor.shape[0]
    if sentinel_20m_tensor.shape[0] != batch_size or landsat_angles.shape[0] != batch_size or sentinel_angles.shape[0] != batch_size:
        raise ValueError("Batch size must be consistent across all input tensors")
    
    # Create RSR reader
    rsr_reader = Landsat8RSRReader(rsr_file_path)
    
    # Print RSR information
    landsat_matrix, wavelengths = rsr_reader.read_rsr_file()
    print(f"Landsat-8 RSR loaded successfully:")
    print(f"  Matrix shape: {landsat_matrix.shape}")
    print(f"  Wavelength range: {wavelengths[0]:.2f} - {wavelengths[-1]:.2f} micrometers")
    
    # Create generator configuration
    config = GeneratorConfig(
        # Input/Output specifications
        landsat_channels=6,
        sentinel_vnir_channels=4,
        sentinel_swir_channels=2,
        angle_dim=4,
        
        # Hyperspectral representation
        landsat_hyperspectral_dim=landsat_matrix.shape[1] if apply_rsr else 150,
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
        
        # Conditional normalization
        use_conditional_norm=use_conditional_norm,
        cond_norm_position="all",
        
        # Physical constraints
        apply_rsr=apply_rsr,
        rsr_position="end",
        rsr_file_path=rsr_file_path
    )
    
    # Print configuration summary
    print(f"\nGenerator configuration:")
    print(f"  Hyperspectral: {config.use_hyperspectral}, Apply RSR: {config.apply_rsr}")
    print(f"  Landsat Hyperspectral dim: {config.landsat_hyperspectral_dim}")
    print(f"  Attention: {config.use_attention}, Conditional norm: {config.use_conditional_norm}")
    print(f"  Residual blocks: {config.num_res_blocks}, Base filters: {config.base_filters}")
    
    # Create the generator
    generator = SentinelToLandsatGenerator(config, rsr_reader)
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
        landsat_img = generator(sentinel_10m_tensor, sentinel_20m_tensor, landsat_angles, sentinel_angles)
    
    # Print output shapes
    print(f"  Sentinel 10m input shape: {sentinel_10m_tensor.shape}")
    print(f"  Sentinel 20m input shape: {sentinel_20m_tensor.shape}")
    print(f"  Landsat output shape: {landsat_img.shape}")
    print(f"  Expected Landsat shape: [B, 6, 128, 128]")
    
    # Verify output ranges
    print(f"\nOutput value ranges:")
    print(f"  Landsat min/max: {landsat_img.min().item():.3f}/{landsat_img.max().item():.3f}")
    print(f"  Expected range: -1.0/1.0 (from tanh activation)")
    
    return landsat_img


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy input tensors
    batch_size = 2
    sentinel_10m_img = torch.randn(batch_size, 4, 384, 384, device=device)  # [B, 4, 384, 384]
    sentinel_20m_img = torch.randn(batch_size, 2, 192, 192, device=device)  # [B, 2, 192, 192]
    landsat_angles = torch.randn(batch_size, 4, device=device)             # [B, 4]
    sentinel_angles = torch.randn(batch_size, 4, device=device)            # [B, 4]
    
    # Path to RSR file
    rsr_file_path = "/home/mrhouma/Documents/Project_perso/SatelliteMapper/data/L8_OLI_RSR.rsr"
    
    # Test with RSR
    print("\n==== Testing with RSR ====")
    landsat_img_rsr = test_sentinel_to_landsat_generator(
        sentinel_10m_img,
        sentinel_20m_img,
        landsat_angles,
        sentinel_angles,
        rsr_file_path,
        base_filters=64,
        num_res_blocks=9,
        use_hyperspectral=True,
        apply_rsr=True,
        use_attention=True,
        use_conditional_norm=True
    )
    
    # Test without RSR
    print("\n==== Testing without RSR ====")
    landsat_img_no_rsr = test_sentinel_to_landsat_generator(
        sentinel_10m_img,
        sentinel_20m_img,
        landsat_angles,
        sentinel_angles,
        rsr_file_path,
        base_filters=64,
        num_res_blocks=9,
        use_hyperspectral=True,
        apply_rsr=False,
        use_attention=True,
        use_conditional_norm=True
    )
    
    # Compare outputs
    print("\nComparing outputs from two configurations:")
    print(f"  With RSR shape: {landsat_img_rsr.shape}")
    print(f"  Without RSR shape: {landsat_img_no_rsr.shape}")
    
    # Check if the outputs are physically sensible
    print("\nPhysical consistency check:")
    print(f"  With RSR - bands min/max: {[f'{landsat_img_rsr[:, i].min().item():.2f}/{landsat_img_rsr[:, i].max().item():.2f}' for i in range(6)]}")
    print(f"  Without RSR - bands min/max: {[f'{landsat_img_no_rsr[:, i].min().item():.2f}/{landsat_img_no_rsr[:, i].max().item():.2f}' for i in range(6)]}")
    
    print("\nGenerator test completed successfully!")