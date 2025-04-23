import torch
import os

def test_landsat8_discriminator(
    landsat_tensor,
    landsat_angles,
    base_filters=64,
    max_filters=512,
    n_layers=3,  # Fewer layers due to smaller input (128×128)
    use_patch_gan=True,
    use_conditional_disc=True,
    use_sigmoid=False
):
    """
    Test the Landsat-8 discriminator with sample inputs.
    
    Args:
        landsat_tensor (torch.Tensor): Landsat-8 image tensor [B, 6, 128, 128]
        landsat_angles (torch.Tensor): Landsat-8 angular information [B, 4]
        base_filters (int): Base number of filters in the discriminator
        max_filters (int): Maximum number of filters in any layer
        n_layers (int): Number of downsampling layers
        use_patch_gan (bool): Whether to use PatchGAN architecture
        use_conditional_disc (bool): Whether to use conditional discrimination with angles
        use_sigmoid (bool): Whether to use sigmoid at the output
        
    Returns:
        torch.Tensor: Discriminator output
    """
    # Validate input tensors
    if landsat_tensor.ndim != 4 or landsat_tensor.shape[1] != 6:
        raise ValueError(f"Expected landsat_tensor with shape [B, 6, 128, 128], got {landsat_tensor.shape}")
    
    if landsat_angles.ndim != 2 or landsat_angles.shape[1] != 4:
        raise ValueError(f"Expected landsat_angles with shape [B, 4], got {landsat_angles.shape}")
    
    # Check batch size consistency
    batch_size = landsat_tensor.shape[0]
    if landsat_angles.shape[0] != batch_size:
        raise ValueError("Batch size must be consistent across all input tensors")
    
    # Get device from input tensor
    device = landsat_tensor.device
    
    # Create discriminator configuration
    from DiscriminatorConfig import DiscriminatorConfig
    
    config = DiscriminatorConfig(
        # Input specifications
        landsat_channels=6,
        angle_dim=4,
        
        # Resolution specifications (from input tensor)
        landsat_resolution=landsat_tensor.shape[2],
        
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
    
    # Create Landsat-8 discriminator
    from Landsat8Discriminator import Landsat8Discriminator
    
    disc = Landsat8Discriminator(config)
    disc = disc.to(device)
    
    # Print model summary (parameter count)
    params = sum(p.numel() for p in disc.parameters())
    print(f"\nModel parameters:")
    print(f"  Landsat-8 discriminator: {params:,}")
    
    # Set model to evaluation mode
    disc.eval()
    
    # Run the discriminator
    print("\nRunning discriminator...")
    with torch.no_grad():
        if use_conditional_disc:
            disc_output = disc(landsat_tensor, landsat_angles)
        else:
            disc_output = disc(landsat_tensor)
    
    # Print output shape
    print(f"  Landsat-8 discriminator output shape: {disc_output.shape}")
    
    # Calculate theoretical output dimensions
    input_size = landsat_tensor.shape[2]
    
    # For PatchGAN, the output size is approximately input_size / (2^n_layers)
    expected_output_size = input_size // (2**n_layers)
    
    print(f"\nExpected output dimensions:")
    print(f"  {batch_size} × 1 × {expected_output_size} × {expected_output_size}")
    
    # Print some statistics about the output values
    print(f"\nOutput value statistics:")
    print(f"  Min/max/mean: {disc_output.min().item():.3f}/{disc_output.max().item():.3f}/{disc_output.mean().item():.3f}")
    
    # For discriminators with sigmoid, values should be between 0 and 1
    if use_sigmoid:
        print(f"  Expected range: 0.0/1.0 (from sigmoid activation)")
    else:
        print(f"  No sigmoid: values can be any real number (typically trained with LSGAN loss)")
    
    return disc_output


# Example usage:
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create dummy input tensors
    batch_size = 2
    landsat_img = torch.randn(batch_size, 6, 128, 128, device=device)  # [B, 6, 128, 128]
    landsat_angles = torch.randn(batch_size, 4, device=device)         # [B, 4]
    
    # Test the discriminator
    landsat_output = test_landsat8_discriminator(
        landsat_img,
        landsat_angles,
        base_filters=64,
        max_filters=512,
        n_layers=3,
        use_patch_gan=True,
        use_conditional_disc=True,
        use_sigmoid=False  # False for LSGAN style
    )
    
    print("\nDiscriminator test completed successfully!")