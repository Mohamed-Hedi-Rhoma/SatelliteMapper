import torch
import os

def test_sentinel2_discriminator(
    sentinel_vnir_tensor,
    sentinel_swir_tensor,
    sentinel_angles,
    base_filters=64,
    max_filters=512,
    n_layers=4,
    use_patch_gan=True,
    use_conditional_disc=True,
    use_sigmoid=False
):
    """
    Test the Sentinel-2 discriminator with sample inputs.
    
    Args:
        sentinel_vnir_tensor (torch.Tensor): Sentinel-2 VNIR image tensor [B, 4, 384, 384]
        sentinel_swir_tensor (torch.Tensor): Sentinel-2 SWIR image tensor [B, 2, 192, 192]
        sentinel_angles (torch.Tensor): Sentinel-2 angular information [B, 4]
        base_filters (int): Base number of filters in the discriminator
        max_filters (int): Maximum number of filters in any layer
        n_layers (int): Number of downsampling layers
        use_patch_gan (bool): Whether to use PatchGAN architecture
        use_conditional_disc (bool): Whether to use conditional discrimination with angles
        use_sigmoid (bool): Whether to use sigmoid at the output
        
    Returns:
        tuple: (vnir_disc_output, swir_disc_output) - Discriminator outputs for VNIR and SWIR
    """
    # Validate input tensors
    if sentinel_vnir_tensor.ndim != 4 or sentinel_vnir_tensor.shape[1] != 4:
        raise ValueError(f"Expected sentinel_vnir_tensor with shape [B, 4, 384, 384], got {sentinel_vnir_tensor.shape}")
    
    if sentinel_swir_tensor.ndim != 4 or sentinel_swir_tensor.shape[1] != 2:
        raise ValueError(f"Expected sentinel_swir_tensor with shape [B, 2, 192, 192], got {sentinel_swir_tensor.shape}")
    
    if sentinel_angles.ndim != 2 or sentinel_angles.shape[1] != 4:
        raise ValueError(f"Expected sentinel_angles with shape [B, 4], got {sentinel_angles.shape}")
    
    # Check batch size consistency
    batch_size = sentinel_vnir_tensor.shape[0]
    if sentinel_swir_tensor.shape[0] != batch_size or sentinel_angles.shape[0] != batch_size:
        raise ValueError("Batch size must be consistent across all input tensors")
    
    # Get device from input tensor
    device = sentinel_vnir_tensor.device
    
    # Create discriminator configuration
    from DiscriminatorConfig import DiscriminatorConfig
    
    config = DiscriminatorConfig(
        # Input specifications
        vnir_channels=4,
        swir_channels=2,
        angle_dim=4,
        
        # Resolution specifications (from input tensors)
        vnir_resolution=sentinel_vnir_tensor.shape[2],
        swir_resolution=sentinel_swir_tensor.shape[2],
        
        # Network architecture
        base_filters=base_filters,
        max_filters=max_filters,
        n_layers=n_layers,
        kernel_size=4,
        
        # Discrimination type
        use_patch_gan=use_patch_gan,
        
        # Conditional discrimination
        use_conditional_disc=use_conditional_disc,
        
        # Normalization and activation
        norm_type="instance",
        use_sigmoid=use_sigmoid
    )
    
    # Print configuration summary
    print(f"\nDiscriminator configuration:")
    print(f"  Base filters: {config.base_filters}, Max filters: {config.max_filters}")
    print(f"  Number of layers: {config.n_layers}")
    print(f"  PatchGAN: {config.use_patch_gan}")
    print(f"  Conditional discrimination: {config.use_conditional_disc}")
    print(f"  Sigmoid output: {config.use_sigmoid}")
    
    # Create VNIR discriminator
    from Sentinel2Discriminator import Sentinel2Discriminator
    
    vnir_disc = Sentinel2Discriminator(config, disc_type='vnir')
    vnir_disc = vnir_disc.to(device)
    
    # Create SWIR discriminator
    swir_disc = Sentinel2Discriminator(config, disc_type='swir')
    swir_disc = swir_disc.to(device)
    
    # Print model summary (parameter count)
    vnir_params = sum(p.numel() for p in vnir_disc.parameters())
    swir_params = sum(p.numel() for p in swir_disc.parameters())
    print(f"\nModel parameters:")
    print(f"  VNIR discriminator: {vnir_params:,}")
    print(f"  SWIR discriminator: {swir_params:,}")
    
    # Set models to evaluation mode
    vnir_disc.eval()
    swir_disc.eval()
    
    # Run the discriminators
    print("\nRunning discriminators...")
    with torch.no_grad():
        if use_conditional_disc:
            vnir_disc_output = vnir_disc(sentinel_vnir_tensor, sentinel_angles)
            swir_disc_output = swir_disc(sentinel_swir_tensor, sentinel_angles)
        else:
            vnir_disc_output = vnir_disc(sentinel_vnir_tensor)
            swir_disc_output = swir_disc(sentinel_swir_tensor)
    
    # Print output shapes
    print(f"  VNIR discriminator output shape: {vnir_disc_output.shape}")
    print(f"  SWIR discriminator output shape: {swir_disc_output.shape}")
    
    # Calculate theoretical output dimensions
    vnir_input_size = sentinel_vnir_tensor.shape[2]
    swir_input_size = sentinel_swir_tensor.shape[2]
    
    # For PatchGAN, the output size is approximately input_size / (2^n_layers)
    expected_vnir_output_size = vnir_input_size // (2**n_layers)
    expected_swir_output_size = swir_input_size // (2**n_layers)
    
    print(f"\nExpected output dimensions:")
    print(f"  VNIR: {batch_size} × 1 × {expected_vnir_output_size} × {expected_vnir_output_size}")
    print(f"  SWIR: {batch_size} × 1 × {expected_swir_output_size} × {expected_swir_output_size}")
    
    # Print some statistics about the output values
    print(f"\nOutput value statistics:")
    print(f"  VNIR min/max/mean: {vnir_disc_output.min().item():.3f}/{vnir_disc_output.max().item():.3f}/{vnir_disc_output.mean().item():.3f}")
    print(f"  SWIR min/max/mean: {swir_disc_output.min().item():.3f}/{swir_disc_output.max().item():.3f}/{swir_disc_output.mean().item():.3f}")
    
    # For discriminators with sigmoid, values should be between 0 and 1
    if use_sigmoid:
        print(f"  Expected range: 0.0/1.0 (from sigmoid activation)")
    else:
        print(f"  No sigmoid: values can be any real number (typically trained with LSGAN loss)")
    
    return vnir_disc_output, swir_disc_output


# Example usage:
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy input tensors
    batch_size = 2
    sentinel_vnir = torch.randn(batch_size, 4, 384, 384, device=device)  # [B, 4, 384, 384]
    sentinel_swir = torch.randn(batch_size, 2, 192, 192, device=device)  # [B, 2, 192, 192]
    sentinel_angles = torch.randn(batch_size, 4, device=device)          # [B, 4]
    
    # Test the discriminators
    vnir_output, swir_output = test_sentinel2_discriminator(
        sentinel_vnir,
        sentinel_swir,
        sentinel_angles,
        base_filters=64,
        max_filters=512,
        n_layers=4,
        use_patch_gan=True,
        use_conditional_disc=True,
        use_sigmoid=False  # False for LSGAN style
    )
    
    print("\nDiscriminator test completed successfully!")