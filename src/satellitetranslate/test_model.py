import torch
import os
import numpy as np

def test_full_cyclegan_architecture(
    landsat_tensor,
    sentinel_10m_tensor,
    sentinel_20m_tensor,
    landsat_angles,
    sentinel_angles,
    landsat_rsr_path,
    sentinel_rsr_path,
    batch_size=2
):
    """
    Test the complete CycleGAN architecture for Landsat-8 and Sentinel-2 translation.
    
    Args:
        landsat_tensor (torch.Tensor): Landsat-8 image tensor [B, 6, 128, 128]
        sentinel_10m_tensor (torch.Tensor): Sentinel-2 10m bands tensor [B, 4, 384, 384]
        sentinel_20m_tensor (torch.Tensor): Sentinel-2 20m bands tensor [B, 2, 192, 192]
        landsat_angles (torch.Tensor): Landsat-8 angular information [B, 4]
        sentinel_angles (torch.Tensor): Sentinel-2 angular information [B, 4]
        landsat_rsr_path (str): Path to the Landsat-8 RSR file
        sentinel_rsr_path (str): Path to the Sentinel-2 RSR file
        batch_size (int): Batch size for all tensors
    """
    # Get device from input tensor
    device = landsat_tensor.device
    print(f"Using device: {device}")
    
    # Validate input tensors
    if landsat_tensor.shape != (batch_size, 6, 128, 128):
        raise ValueError(f"Expected landsat_tensor with shape [{batch_size}, 6, 128, 128], got {landsat_tensor.shape}")
    
    if sentinel_10m_tensor.shape != (batch_size, 4, 384, 384):
        raise ValueError(f"Expected sentinel_10m_tensor with shape [{batch_size}, 4, 384, 384], got {sentinel_10m_tensor.shape}")
    
    if sentinel_20m_tensor.shape != (batch_size, 2, 192, 192):
        raise ValueError(f"Expected sentinel_20m_tensor with shape [{batch_size}, 2, 192, 192], got {sentinel_20m_tensor.shape}")
    
    print("\n===== STEP 1: TESTING GENERATORS =====")
    
    # Import generator classes and config
    from generatorconfig import GeneratorConfig
    from LandsatToSentinelGenerator import LandsatToSentinelGenerator
    from SentinelToLandsatGenerator import SentinelToLandsatGenerator
    from utils.sentinel_rsr import RSRReader
    from utils.landsat8_rsr import Landsat8RSRReader
    
    # Create RSR readers
    sentinel_rsr_reader = RSRReader(sentinel_rsr_path)
    landsat_rsr_reader = Landsat8RSRReader(landsat_rsr_path)
    
    # Create generator configurations
    print("\nCreating Landsat→Sentinel Generator config...")
    l8_to_s2_config = GeneratorConfig(
        # Input/Output specifications
        landsat_channels=6,
        sentinel_vnir_channels=4,
        sentinel_swir_channels=2,
        angle_dim=4,
        
        # Hyperspectral representation
        use_hyperspectral=True,
        
        # Network architecture
        base_filters=64,
        num_res_blocks=9,
        use_attention=True,
        norm_type="instance",
        
        # Upsampling specifications
        vnir_upsampling_factor=3,
        swir_upsampling_factor=1.5,
        
        # Conditional normalization
        use_conditional_norm=True,
        
        # Physical constraints
        apply_rsr=True,
        
        # Multi-resolution decoder options
        separate_decoders=True
    )
    
    print("\nCreating Sentinel→Landsat Generator config...")
    s2_to_l8_config = GeneratorConfig(
        # Input/Output specifications
        landsat_channels=6,
        sentinel_vnir_channels=4,
        sentinel_swir_channels=2,
        angle_dim=4,
        landsat_hyperspectral_dim=0,  # Will be set from RSR
        
        # Hyperspectral representation
        use_hyperspectral=True,
        
        # Network architecture
        base_filters=64,
        num_res_blocks=9,
        use_attention=True,
        norm_type="instance",
        
        # Conditional normalization
        use_conditional_norm=True,
        
        # Physical constraints
        apply_rsr=True
    )
    
    # Create generators
    print("\nInitializing Landsat→Sentinel Generator...")
    l8_to_s2_gen = LandsatToSentinelGenerator(l8_to_s2_config, sentinel_rsr_reader)
    l8_to_s2_gen = l8_to_s2_gen.to(device)
    
    print("\nInitializing Sentinel→Landsat Generator...")
    s2_to_l8_gen = SentinelToLandsatGenerator(s2_to_l8_config, landsat_rsr_reader)
    s2_to_l8_gen = s2_to_l8_gen.to(device)
    
    # Set models to evaluation mode
    l8_to_s2_gen.eval()
    s2_to_l8_gen.eval()
    
    # Test forward passes
    print("\nTesting Landsat→Sentinel Generator forward pass...")
    with torch.no_grad():
        fake_sentinel_10m, fake_sentinel_20m = l8_to_s2_gen(landsat_tensor, sentinel_angles, landsat_angles)
    
    print("\nTesting Sentinel→Landsat Generator forward pass...")
    with torch.no_grad():
        fake_landsat = s2_to_l8_gen(sentinel_10m_tensor, sentinel_20m_tensor, landsat_angles, sentinel_angles)
    
    # Print generator output shapes
    print("\nGenerator output shapes:")
    print(f"  Fake Sentinel 10m: {fake_sentinel_10m.shape}")
    print(f"  Fake Sentinel 20m: {fake_sentinel_20m.shape}")
    print(f"  Fake Landsat: {fake_landsat.shape}")
    
    # Test cycle consistency
    print("\nTesting CycleGAN cycle consistency...")
    with torch.no_grad():
        # Landsat → Sentinel → Landsat
        reconstructed_landsat = s2_to_l8_gen(fake_sentinel_10m, fake_sentinel_20m, landsat_angles, sentinel_angles)
        
        # Sentinel → Landsat → Sentinel
        reconstructed_sentinel_10m, reconstructed_sentinel_20m = l8_to_s2_gen(fake_landsat, sentinel_angles, landsat_angles)
    
    # Print reconstruction shapes
    print("\nReconstruction output shapes:")
    print(f"  Reconstructed Landsat: {reconstructed_landsat.shape}")
    print(f"  Reconstructed Sentinel 10m: {reconstructed_sentinel_10m.shape}")
    print(f"  Reconstructed Sentinel 20m: {reconstructed_sentinel_20m.shape}")
    
    # Calculate L1 cycle consistency losses
    landsat_cycle_loss = torch.abs(reconstructed_landsat - landsat_tensor).mean()
    sentinel_10m_cycle_loss = torch.abs(reconstructed_sentinel_10m - sentinel_10m_tensor).mean()
    sentinel_20m_cycle_loss = torch.abs(reconstructed_sentinel_20m - sentinel_20m_tensor).mean()
    
    print("\nCycle consistency losses (L1):")
    print(f"  Landsat cycle: {landsat_cycle_loss.item():.4f}")
    print(f"  Sentinel 10m cycle: {sentinel_10m_cycle_loss.item():.4f}")
    print(f"  Sentinel 20m cycle: {sentinel_20m_cycle_loss.item():.4f}")
    
    print("\n===== STEP 2: TESTING DISCRIMINATORS =====")
    
    # Import discriminator classes and config
    from DiscriminatorConfig import DiscriminatorConfig
    from Sentinel2Discriminator import Sentinel2Discriminator
    from Landsat8Discriminator import Landsat8Discriminator
    
    # Create discriminator configuration
    disc_config = DiscriminatorConfig(
        # Input specifications
        vnir_channels=4,
        swir_channels=2,
        landsat_channels=6,
        angle_dim=4,
        
        # Resolution specifications
        vnir_resolution=384,
        swir_resolution=192,
        landsat_resolution=128,
        
        # Network architecture
        base_filters=64,
        max_filters=512,
        n_layers=4,
        
        # Discrimination type
        use_patch_gan=True,
        
        # Conditional discrimination
        use_conditional_disc=True,
        
        # Normalization and activation
        norm_type="instance",
        use_sigmoid=False
    )
    
    # Create discriminators
    print("\nInitializing Sentinel-2 VNIR Discriminator...")
    s2_vnir_disc = Sentinel2Discriminator(disc_config, disc_type='vnir')
    s2_vnir_disc = s2_vnir_disc.to(device)
    
    print("\nInitializing Sentinel-2 SWIR Discriminator...")
    s2_swir_disc = Sentinel2Discriminator(disc_config, disc_type='swir')
    s2_swir_disc = s2_swir_disc.to(device)
    
    print("\nInitializing Landsat-8 Discriminator...")
    l8_disc = Landsat8Discriminator(disc_config)
    l8_disc = l8_disc.to(device)
    
    # Set models to evaluation mode
    s2_vnir_disc.eval()
    s2_swir_disc.eval()
    l8_disc.eval()
    
    # Test discriminator forward passes
    print("\nTesting Sentinel-2 VNIR Discriminator (real data)...")
    with torch.no_grad():
        real_s2_vnir_score = s2_vnir_disc(sentinel_10m_tensor, sentinel_angles)
    
    print("\nTesting Sentinel-2 SWIR Discriminator (real data)...")
    with torch.no_grad():
        real_s2_swir_score = s2_swir_disc(sentinel_20m_tensor, sentinel_angles)
    
    print("\nTesting Landsat-8 Discriminator (real data)...")
    with torch.no_grad():
        real_l8_score = l8_disc(landsat_tensor, landsat_angles)
    
    # Test with fake data
    print("\nTesting Sentinel-2 VNIR Discriminator (fake data)...")
    with torch.no_grad():
        fake_s2_vnir_score = s2_vnir_disc(fake_sentinel_10m, sentinel_angles)
    
    print("\nTesting Sentinel-2 SWIR Discriminator (fake data)...")
    with torch.no_grad():
        fake_s2_swir_score = s2_swir_disc(fake_sentinel_20m, sentinel_angles)
    
    print("\nTesting Landsat-8 Discriminator (fake data)...")
    with torch.no_grad():
        fake_l8_score = l8_disc(fake_landsat, landsat_angles)
    
    # Print discriminator output shapes
    print("\nDiscriminator output shapes (real data):")
    print(f"  Sentinel-2 VNIR: {real_s2_vnir_score.shape}")
    print(f"  Sentinel-2 SWIR: {real_s2_swir_score.shape}")
    print(f"  Landsat-8: {real_l8_score.shape}")
    
    # Print discriminator score statistics
    print("\nDiscriminator scores (real data):")
    print(f"  Sentinel-2 VNIR - min/max/mean: {real_s2_vnir_score.min().item():.3f}/{real_s2_vnir_score.max().item():.3f}/{real_s2_vnir_score.mean().item():.3f}")
    print(f"  Sentinel-2 SWIR - min/max/mean: {real_s2_swir_score.min().item():.3f}/{real_s2_swir_score.max().item():.3f}/{real_s2_swir_score.mean().item():.3f}")
    print(f"  Landsat-8 - min/max/mean: {real_l8_score.min().item():.3f}/{real_l8_score.max().item():.3f}/{real_l8_score.mean().item():.3f}")
    
    print("\nDiscriminator scores (fake data):")
    print(f"  Sentinel-2 VNIR - min/max/mean: {fake_s2_vnir_score.min().item():.3f}/{fake_s2_vnir_score.max().item():.3f}/{fake_s2_vnir_score.mean().item():.3f}")
    print(f"  Sentinel-2 SWIR - min/max/mean: {fake_s2_swir_score.min().item():.3f}/{fake_s2_swir_score.max().item():.3f}/{fake_s2_swir_score.mean().item():.3f}")
    print(f"  Landsat-8 - min/max/mean: {fake_l8_score.min().item():.3f}/{fake_l8_score.max().item():.3f}/{fake_l8_score.mean().item():.3f}")
    
    print("\n===== STEP 3: TESTING FULL CYCLEGAN LOSSES =====")
    
    # Calculate GAN losses
    # For LSGAN style:
    # - Target for real: 1.0
    # - Target for fake: 0.0
    mse_loss = torch.nn.MSELoss()
    
    # Generator losses
    g_l8_to_s2_loss_vnir = mse_loss(fake_s2_vnir_score, torch.ones_like(fake_s2_vnir_score))
    g_l8_to_s2_loss_swir = mse_loss(fake_s2_swir_score, torch.ones_like(fake_s2_swir_score))
    g_s2_to_l8_loss = mse_loss(fake_l8_score, torch.ones_like(fake_l8_score))
    
    # Total generator loss (plus cycle consistency)
    lambda_cycle = 10.0  # Weight for cycle consistency losses
    
    g_total_loss = (
        g_l8_to_s2_loss_vnir + 
        g_l8_to_s2_loss_swir + 
        g_s2_to_l8_loss + 
        lambda_cycle * (landsat_cycle_loss + sentinel_10m_cycle_loss + sentinel_20m_cycle_loss)
    )
    
    # Discriminator losses
    d_s2_vnir_real_loss = mse_loss(real_s2_vnir_score, torch.ones_like(real_s2_vnir_score))
    d_s2_vnir_fake_loss = mse_loss(fake_s2_vnir_score, torch.zeros_like(fake_s2_vnir_score))
    d_s2_vnir_loss = 0.5 * (d_s2_vnir_real_loss + d_s2_vnir_fake_loss)
    
    d_s2_swir_real_loss = mse_loss(real_s2_swir_score, torch.ones_like(real_s2_swir_score))
    d_s2_swir_fake_loss = mse_loss(fake_s2_swir_score, torch.zeros_like(fake_s2_swir_score))
    d_s2_swir_loss = 0.5 * (d_s2_swir_real_loss + d_s2_swir_fake_loss)
    
    d_l8_real_loss = mse_loss(real_l8_score, torch.ones_like(real_l8_score))
    d_l8_fake_loss = mse_loss(fake_l8_score, torch.zeros_like(fake_l8_score))
    d_l8_loss = 0.5 * (d_l8_real_loss + d_l8_fake_loss)
    
    d_total_loss = d_s2_vnir_loss + d_s2_swir_loss + d_l8_loss
    
    print("\nGenerator Losses:")
    print(f"  G_L8→S2 (VNIR): {g_l8_to_s2_loss_vnir.item():.4f}")
    print(f"  G_L8→S2 (SWIR): {g_l8_to_s2_loss_swir.item():.4f}")
    print(f"  G_S2→L8: {g_s2_to_l8_loss.item():.4f}")
    print(f"  Cycle Landsat: {landsat_cycle_loss.item():.4f}")
    print(f"  Cycle Sentinel 10m: {sentinel_10m_cycle_loss.item():.4f}")
    print(f"  Cycle Sentinel 20m: {sentinel_20m_cycle_loss.item():.4f}")
    print(f"  G Total Loss: {g_total_loss.item():.4f}")
    
    print("\nDiscriminator Losses:")
    print(f"  D_S2_VNIR: {d_s2_vnir_loss.item():.4f} (Real: {d_s2_vnir_real_loss.item():.4f}, Fake: {d_s2_vnir_fake_loss.item():.4f})")
    print(f"  D_S2_SWIR: {d_s2_swir_loss.item():.4f} (Real: {d_s2_swir_real_loss.item():.4f}, Fake: {d_s2_swir_fake_loss.item():.4f})")
    print(f"  D_L8: {d_l8_loss.item():.4f} (Real: {d_l8_real_loss.item():.4f}, Fake: {d_l8_fake_loss.item():.4f})")
    print(f"  D Total Loss: {d_total_loss.item():.4f}")
    
    print("\n===== FULL CYCLEGAN ARCHITECTURE TEST COMPLETED SUCCESSFULLY =====")
    
    return {
        'generators': {
            'l8_to_s2': l8_to_s2_gen,
            's2_to_l8': s2_to_l8_gen
        },
        'discriminators': {
            's2_vnir': s2_vnir_disc,
            's2_swir': s2_swir_disc,
            'l8': l8_disc
        },
        'outputs': {
            'fake_sentinel_10m': fake_sentinel_10m,
            'fake_sentinel_20m': fake_sentinel_20m,
            'fake_landsat': fake_landsat,
            'reconstructed_landsat': reconstructed_landsat,
            'reconstructed_sentinel_10m': reconstructed_sentinel_10m,
            'reconstructed_sentinel_20m': reconstructed_sentinel_20m
        },
        'losses': {
            'g_loss': g_total_loss.item(),
            'd_loss': d_total_loss.item(),
            'cycle_losses': {
                'landsat': landsat_cycle_loss.item(),
                'sentinel_10m': sentinel_10m_cycle_loss.item(),
                'sentinel_20m': sentinel_20m_cycle_loss.item()
            }
        }
    }


# Example usage
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dummy input tensors
    batch_size = 2
    landsat_img = torch.randn(batch_size, 6, 128, 128, device=device)  # [B, 6, 128, 128]
    sentinel_10m_img = torch.randn(batch_size, 4, 384, 384, device=device)  # [B, 4, 384, 384]
    sentinel_20m_img = torch.randn(batch_size, 2, 192, 192, device=device)  # [B, 2, 192, 192]
    landsat_angles = torch.randn(batch_size, 4, device=device)          # [B, 4]
    sentinel_angles = torch.randn(batch_size, 4, device=device)         # [B, 4]
    
    # Set paths to RSR files (modify these to your actual paths)
    landsat_rsr_path = "/home/mrhouma/Documents/Project_perso/SatelliteMapper/data/L8_OLI_RSR.rsr"
    sentinel_rsr_path = "/home/mrhouma/Documents/Project_perso/SatelliteMapper/data/sentinel2.rsr"
    
    # Test full CycleGAN architecture
    results = test_full_cyclegan_architecture(
        landsat_img,
        sentinel_10m_img,
        sentinel_20m_img,
        landsat_angles,
        sentinel_angles,
        landsat_rsr_path,
        sentinel_rsr_path,
        batch_size=batch_size
    )