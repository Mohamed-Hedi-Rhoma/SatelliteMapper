import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import your custom modules
from satellitetranslate.generatorconfig import GeneratorConfig
from satellitetranslate.DiscriminatorConfig import DiscriminatorConfig
from satellitetranslate.LandsatToSentinelGenerator import LandsatToSentinelGenerator
from satellitetranslate.SentinelToLandsatGenerator import SentinelToLandsatGenerator
from satellitetranslate.Sentinel2Discriminator import Sentinel2Discriminator
from satellitetranslate.Landsat8Discriminator import Landsat8Discriminator
from satellitetranslate.utils.sentinel_rsr import RSRReader
from satellitetranslate.utils.landsat8_rsr import Landsat8RSRReader
from dataset.dataloader import create_train_dataloader, create_valid_dataloader


def train_cyclegan(
    tensor_dir,
    landsat_rsr_path,
    sentinel_rsr_path,
    output_dir,
    num_epochs=10,
    batch_size=4,
    lr=0.0002,
    beta1=0.5,
    lambda_cycle=10.0,
    lambda_identity=0.5,
    save_interval=1
):
    """
    Train the physics-informed CycleGAN for Landsat-8 and Sentinel-2 translation.
    
    Args:
        tensor_dir: Directory containing tensor and angle files
        landsat_rsr_path: Path to the Landsat-8 RSR file
        sentinel_rsr_path: Path to the Sentinel-2 RSR file
        output_dir: Directory to save model checkpoints and logs
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        lr: Learning rate
        beta1: Beta1 parameter for Adam optimizer
        lambda_cycle: Weight for cycle consistency loss
        lambda_identity: Weight for identity loss
        save_interval: Epoch interval to save model checkpoints
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader = create_train_dataloader(tensor_dir, batch_size=batch_size)
    valid_loader = create_valid_dataloader(tensor_dir, batch_size=batch_size)
    
    # Create RSR readers
    sentinel_rsr_reader = RSRReader(sentinel_rsr_path)
    landsat_rsr_reader = Landsat8RSRReader(landsat_rsr_path)
    
    # Create generator configurations
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
        use_sigmoid=False  # Use LSGAN
    )
    
    # Initialize models
    print("Initializing models...")
    
    # Generators
    G_L8_to_S2 = LandsatToSentinelGenerator(l8_to_s2_config, sentinel_rsr_reader).to(device)
    G_S2_to_L8 = SentinelToLandsatGenerator(s2_to_l8_config, landsat_rsr_reader).to(device)
    
    # Discriminators
    D_S2_VNIR = Sentinel2Discriminator(disc_config, disc_type='vnir').to(device)
    D_S2_SWIR = Sentinel2Discriminator(disc_config, disc_type='swir').to(device)
    D_L8 = Landsat8Discriminator(disc_config).to(device)
    
    # Initialize optimizers
    optimizer_G = optim.Adam(
        list(G_L8_to_S2.parameters()) + list(G_S2_to_L8.parameters()),
        lr=lr,
        betas=(beta1, 0.999)
    )
    
    optimizer_D = optim.Adam(
        list(D_S2_VNIR.parameters()) + list(D_S2_SWIR.parameters()) + list(D_L8.parameters()),
        lr=lr,
        betas=(beta1, 0.999)
    )
    
    # Loss functions
    criterionGAN = nn.MSELoss()  # LSGAN
    criterionCycle = nn.L1Loss()
    criterionIdentity = nn.L1Loss()
    
    # Learning rate schedulers - decrease LR by half every 5 epochs
    scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=5, gamma=0.5)
    scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=5, gamma=0.5)
    
    # Training history
    history = {
        'G_loss': [],
        'D_loss': [],
        'G_cycle_loss': [],
        'G_identity_loss': [],
        'G_adv_loss': [],
        'D_S2_VNIR_loss': [],
        'D_S2_SWIR_loss': [],
        'D_L8_loss': []
    }
    
    # Function to set requires_grad
    def set_requires_grad(nets, requires_grad=False):
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    
    # Get a sample batch for visualization
    val_batch = next(iter(valid_loader))
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training metrics
        epoch_G_loss = 0.0
        epoch_D_loss = 0.0
        epoch_G_cycle_loss = 0.0
        epoch_G_identity_loss = 0.0
        epoch_G_adv_loss = 0.0
        epoch_D_S2_VNIR_loss = 0.0
        epoch_D_S2_SWIR_loss = 0.0
        epoch_D_L8_loss = 0.0
        
        # Set models to training mode
        G_L8_to_S2.train()
        G_S2_to_L8.train()
        D_S2_VNIR.train()
        D_S2_SWIR.train()
        D_L8.train()
        
        # Wrap the loader with tqdm for a progress bar
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for i, batch in enumerate(train_bar):
            # Get real data
            real_L8 = batch['landsat_img'].to(device)
            real_S2_VNIR = batch['sentinel_10m_img'].to(device)
            real_S2_SWIR = batch['sentinel_20m_img'].to(device)
            L8_angles = batch['landsat_angles'].to(device)
            S2_angles = batch['sentinel_angles'].to(device)
            
            # Real target values (1.0 for real, 0.0 for fake)
            real_label = torch.ones(real_L8.size(0), 1, 11, 11).to(device)  # Size for PatchGAN
            fake_label = torch.zeros(real_L8.size(0), 1, 11, 11).to(device)
            
            # Adjust label size for Landsat discriminator
            real_label_L8 = torch.ones(real_L8.size(0), 1, 15, 15).to(device)
            fake_label_L8 = torch.zeros(real_L8.size(0), 1, 15, 15).to(device)
            
            # -----------------
            # Train Generators
            # -----------------
            
            # Enable gradients for generators, disable for discriminators
            set_requires_grad([D_S2_VNIR, D_S2_SWIR, D_L8], False)
            
            optimizer_G.zero_grad()
            
            # Identity loss
            if lambda_identity > 0:
                # G_L8_to_S2 should generate same S2 images from real S2
                same_S2_VNIR, same_S2_SWIR = G_L8_to_S2(G_S2_to_L8(real_S2_VNIR, real_S2_SWIR, L8_angles, S2_angles), 
                                                        S2_angles, L8_angles)
                loss_identity_S2_VNIR = criterionIdentity(same_S2_VNIR, real_S2_VNIR) * lambda_identity
                loss_identity_S2_SWIR = criterionIdentity(same_S2_SWIR, real_S2_SWIR) * lambda_identity
                
                # G_S2_to_L8 should generate same L8 images from real L8
                same_L8 = G_S2_to_L8(*G_L8_to_S2(real_L8, S2_angles, L8_angles), L8_angles, S2_angles)
                loss_identity_L8 = criterionIdentity(same_L8, real_L8) * lambda_identity
                
                loss_identity = loss_identity_S2_VNIR + loss_identity_S2_SWIR + loss_identity_L8
            else:
                loss_identity = 0
            
            # GAN loss for G_L8_to_S2
            fake_S2_VNIR, fake_S2_SWIR = G_L8_to_S2(real_L8, S2_angles, L8_angles)
            pred_fake_S2_VNIR = D_S2_VNIR(fake_S2_VNIR, S2_angles)
            pred_fake_S2_SWIR = D_S2_SWIR(fake_S2_SWIR, S2_angles)
            
            loss_GAN_L8_to_S2_VNIR = criterionGAN(pred_fake_S2_VNIR, real_label)
            loss_GAN_L8_to_S2_SWIR = criterionGAN(pred_fake_S2_SWIR, real_label)
            
            # GAN loss for G_S2_to_L8
            fake_L8 = G_S2_to_L8(real_S2_VNIR, real_S2_SWIR, L8_angles, S2_angles)
            pred_fake_L8 = D_L8(fake_L8, L8_angles)
            
            loss_GAN_S2_to_L8 = criterionGAN(pred_fake_L8, real_label_L8)
            
            # Cycle consistency loss
            # L8 -> S2 -> L8
            rec_L8 = G_S2_to_L8(fake_S2_VNIR, fake_S2_SWIR, L8_angles, S2_angles)
            loss_cycle_L8 = criterionCycle(rec_L8, real_L8) * lambda_cycle
            
            # S2 -> L8 -> S2
            rec_S2_VNIR, rec_S2_SWIR = G_L8_to_S2(fake_L8, S2_angles, L8_angles)
            loss_cycle_S2_VNIR = criterionCycle(rec_S2_VNIR, real_S2_VNIR) * lambda_cycle
            loss_cycle_S2_SWIR = criterionCycle(rec_S2_SWIR, real_S2_SWIR) * lambda_cycle
            
            # Total cycle loss
            loss_cycle = loss_cycle_L8 + loss_cycle_S2_VNIR + loss_cycle_S2_SWIR
            
            # Total generator loss
            loss_G_adv = loss_GAN_L8_to_S2_VNIR + loss_GAN_L8_to_S2_SWIR + loss_GAN_S2_to_L8
            loss_G = loss_G_adv + loss_cycle + loss_identity
            
            # Backpropagate and update generators
            loss_G.backward()
            optimizer_G.step()
            
            # -----------------
            # Train Discriminators
            # -----------------
            
            # Enable gradients for discriminators
            set_requires_grad([D_S2_VNIR, D_S2_SWIR, D_L8], True)
            
            optimizer_D.zero_grad()
            
            # D_S2_VNIR loss
            pred_real_S2_VNIR = D_S2_VNIR(real_S2_VNIR, S2_angles)
            loss_D_real_S2_VNIR = criterionGAN(pred_real_S2_VNIR, real_label)
            
            # Detach to avoid backprop through generator
            pred_fake_S2_VNIR = D_S2_VNIR(fake_S2_VNIR.detach(), S2_angles)
            loss_D_fake_S2_VNIR = criterionGAN(pred_fake_S2_VNIR, fake_label)
            
            loss_D_S2_VNIR = (loss_D_real_S2_VNIR + loss_D_fake_S2_VNIR) * 0.5
            
            # D_S2_SWIR loss
            pred_real_S2_SWIR = D_S2_SWIR(real_S2_SWIR, S2_angles)
            loss_D_real_S2_SWIR = criterionGAN(pred_real_S2_SWIR, real_label)
            
            pred_fake_S2_SWIR = D_S2_SWIR(fake_S2_SWIR.detach(), S2_angles)
            loss_D_fake_S2_SWIR = criterionGAN(pred_fake_S2_SWIR, fake_label)
            
            loss_D_S2_SWIR = (loss_D_real_S2_SWIR + loss_D_fake_S2_SWIR) * 0.5
            
            # D_L8 loss
            pred_real_L8 = D_L8(real_L8, L8_angles)
            loss_D_real_L8 = criterionGAN(pred_real_L8, real_label_L8)
            
            pred_fake_L8 = D_L8(fake_L8.detach(), L8_angles)
            loss_D_fake_L8 = criterionGAN(pred_fake_L8, fake_label_L8)
            
            loss_D_L8 = (loss_D_real_L8 + loss_D_fake_L8) * 0.5
            
            # Total discriminator loss
            loss_D = loss_D_S2_VNIR + loss_D_S2_SWIR + loss_D_L8
            
            # Backpropagate and update discriminators
            loss_D.backward()
            optimizer_D.step()
            
            # Update training metrics
            epoch_G_loss += loss_G.item()
            epoch_D_loss += loss_D.item()
            epoch_G_cycle_loss += loss_cycle.item()
            epoch_G_identity_loss += loss_identity.item() if isinstance(loss_identity, torch.Tensor) else loss_identity
            epoch_G_adv_loss += loss_G_adv.item()
            epoch_D_S2_VNIR_loss += loss_D_S2_VNIR.item()
            epoch_D_S2_SWIR_loss += loss_D_S2_SWIR.item()
            epoch_D_L8_loss += loss_D_L8.item()
            
            # Update progress bar
            train_bar.set_postfix({
                'G_loss': loss_G.item(),
                'D_loss': loss_D.item(),
                'Cycle': loss_cycle.item()
            })
        
        # Average metrics over the epoch
        num_batches = len(train_loader)
        epoch_G_loss /= num_batches
        epoch_D_loss /= num_batches
        epoch_G_cycle_loss /= num_batches
        epoch_G_identity_loss /= num_batches
        epoch_G_adv_loss /= num_batches
        epoch_D_S2_VNIR_loss /= num_batches
        epoch_D_S2_SWIR_loss /= num_batches
        epoch_D_L8_loss /= num_batches
        
        # Step the learning rate schedulers
        scheduler_G.step()
        scheduler_D.step()
        
        # Update history
        history['G_loss'].append(epoch_G_loss)
        history['D_loss'].append(epoch_D_loss)
        history['G_cycle_loss'].append(epoch_G_cycle_loss)
        history['G_identity_loss'].append(epoch_G_identity_loss)
        history['G_adv_loss'].append(epoch_G_adv_loss)
        history['D_S2_VNIR_loss'].append(epoch_D_S2_VNIR_loss)
        history['D_S2_SWIR_loss'].append(epoch_D_S2_SWIR_loss)
        history['D_L8_loss'].append(epoch_D_L8_loss)
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"  G_loss: {epoch_G_loss:.4f}, D_loss: {epoch_D_loss:.4f}")
        print(f"  Cycle: {epoch_G_cycle_loss:.4f}, Identity: {epoch_G_identity_loss:.4f}, G_Adv: {epoch_G_adv_loss:.4f}")
        print(f"  D_S2_VNIR: {epoch_D_S2_VNIR_loss:.4f}, D_S2_SWIR: {epoch_D_S2_SWIR_loss:.4f}, D_L8: {epoch_D_L8_loss:.4f}")
        
        # Visualization on validation set
        if (epoch + 1) % save_interval == 0:
            # Set models to evaluation mode
            G_L8_to_S2.eval()
            G_S2_to_L8.eval()
            
            with torch.no_grad():
                # Get validation batch
                val_L8 = val_batch['landsat_img'].to(device)
                val_S2_VNIR = val_batch['sentinel_10m_img'].to(device)
                val_S2_SWIR = val_batch['sentinel_20m_img'].to(device)
                val_L8_angles = val_batch['landsat_angles'].to(device)
                val_S2_angles = val_batch['sentinel_angles'].to(device)
                
                # Generate fake images
                fake_S2_VNIR, fake_S2_SWIR = G_L8_to_S2(val_L8, val_S2_angles, val_L8_angles)
                fake_L8 = G_S2_to_L8(val_S2_VNIR, val_S2_SWIR, val_L8_angles, val_S2_angles)
                
                # Save model checkpoints
                checkpoint_dir = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                torch.save(G_L8_to_S2.state_dict(), os.path.join(checkpoint_dir, 'G_L8_to_S2.pth'))
                torch.save(G_S2_to_L8.state_dict(), os.path.join(checkpoint_dir, 'G_S2_to_L8.pth'))
                torch.save(D_S2_VNIR.state_dict(), os.path.join(checkpoint_dir, 'D_S2_VNIR.pth'))
                torch.save(D_S2_SWIR.state_dict(), os.path.join(checkpoint_dir, 'D_S2_SWIR.pth'))
                torch.save(D_L8.state_dict(), os.path.join(checkpoint_dir, 'D_L8.pth'))
                
                # Save optimizer states
                torch.save(optimizer_G.state_dict(), os.path.join(checkpoint_dir, 'optimizer_G.pth'))
                torch.save(optimizer_D.state_dict(), os.path.join(checkpoint_dir, 'optimizer_D.pth'))
                
                # Visualize results (save sample images)
                visualize_results(val_L8, val_S2_VNIR, val_S2_SWIR, fake_L8, fake_S2_VNIR, fake_S2_SWIR, 
                                 os.path.join(checkpoint_dir, 'samples.png'))
    
    # Plot training history
    plot_training_history(history, os.path.join(output_dir, 'training_history.png'))
    
    # Save final models
    final_checkpoint_dir = os.path.join(output_dir, "final_models")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    torch.save(G_L8_to_S2.state_dict(), os.path.join(final_checkpoint_dir, 'G_L8_to_S2.pth'))
    torch.save(G_S2_to_L8.state_dict(), os.path.join(final_checkpoint_dir, 'G_S2_to_L8.pth'))
    torch.save(D_S2_VNIR.state_dict(), os.path.join(final_checkpoint_dir, 'D_S2_VNIR.pth'))
    torch.save(D_S2_SWIR.state_dict(), os.path.join(final_checkpoint_dir, 'D_S2_SWIR.pth'))
    torch.save(D_L8.state_dict(), os.path.join(final_checkpoint_dir, 'D_L8.pth'))
    
    # Save training history
    save_training_history(history, os.path.join(output_dir, 'training_history.json'))
    
    print("Training completed successfully!")
    return history


def visualize_results(real_L8, real_S2_VNIR, real_S2_SWIR, fake_L8, fake_S2_VNIR, fake_S2_SWIR, save_path):
    """
    Visualize and save sample results from the model.
    
    Args:
        real_L8: Real Landsat-8 images
        real_S2_VNIR: Real Sentinel-2 VNIR images
        real_S2_SWIR: Real Sentinel-2 SWIR images
        fake_L8: Generated Landsat-8 images
        fake_S2_VNIR: Generated Sentinel-2 VNIR images
        fake_S2_SWIR: Generated Sentinel-2 SWIR images
        save_path: Path to save the visualization
    """
    # Convert tensors to numpy arrays
    real_L8 = real_L8.cpu().numpy()
    real_S2_VNIR = real_S2_VNIR.cpu().numpy()
    real_S2_SWIR = real_S2_SWIR.cpu().numpy()
    fake_L8 = fake_L8.cpu().numpy()
    fake_S2_VNIR = fake_S2_VNIR.cpu().numpy()
    fake_S2_SWIR = fake_S2_SWIR.cpu().numpy()
    
    # Get the first image from the batch
    sample_idx = 0
    
    # Normalize images to [0, 1] range for visualization
    def normalize_for_display(img):
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        img = (img + 1) / 2.0  # [-1, 1] -> [0, 1]
        img = np.clip(img, 0, 1)
        return img
    
    # Create RGB composites
    # Landsat-8: Bands 4,3,2 = NIR, Red, Green (SWIR bands are 5,6)
    real_L8_rgb = normalize_for_display(real_L8[sample_idx, [3, 2, 1], :, :])
    fake_L8_rgb = normalize_for_display(fake_L8[sample_idx, [3, 2, 1], :, :])
    
    # Sentinel-2 VNIR: Bands 3,2,1 = NIR, Red, Green 
    real_S2_rgb = normalize_for_display(real_S2_VNIR[sample_idx, [2, 1, 0], :, :])
    fake_S2_rgb = normalize_for_display(fake_S2_VNIR[sample_idx, [2, 1, 0], :, :])
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot images
    axes[0, 0].imshow(real_L8_rgb)
    axes[0, 0].set_title('Real Landsat-8 (NIR-R-G)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(fake_L8_rgb)
    axes[0, 1].set_title('Generated Landsat-8 (NIR-R-G)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(real_S2_rgb)
    axes[1, 0].set_title('Real Sentinel-2 (NIR-R-G)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(fake_S2_rgb)
    axes[1, 1].set_title('Generated Sentinel-2 (NIR-R-G)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(history, save_path):
    """
    Plot and save training history.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot
    """
    epochs = range(1, len(history['G_loss']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Generator losses
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['G_loss'], 'b-', label='G Total Loss')
    plt.plot(epochs, history['G_cycle_loss'], 'r-', label='Cycle Loss')
    plt.plot(epochs, history['G_identity_loss'], 'g-', label='Identity Loss')
    plt.plot(epochs, history['G_adv_loss'], 'm-', label='G Adversarial Loss')
    plt.title('Generator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Discriminator losses
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['D_loss'], 'b-', label='D Total Loss')
    plt.plot(epochs, history['D_S2_VNIR_loss'], 'r-', label='D S2 VNIR Loss')
    plt.plot(epochs, history['D_S2_SWIR_loss'], 'g-', label='D S2 SWIR Loss')
    plt.plot(epochs, history['D_L8_loss'], 'm-', label='D L8 Loss')
    plt.title('Discriminator Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # G vs D loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs, history['G_loss'], 'b-', label='Generator Loss')
    plt.plot(epochs, history['D_loss'], 'r-', label='Discriminator Loss')
    plt.title('Generator vs Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_history(history, save_path):
    """
    Save training history to a JSON file.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the JSON file
    """
    import json
    
    # Convert numpy arrays or tensors to Python lists
    serializable_history = {}
    for key, values in history.items():
        serializable_history[key] = [float(val) for val in values]
    
    # Save to JSON
    with open(save_path, 'w') as f:
        json.dump(serializable_history, f, indent=4)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Physics-Informed CycleGAN for Landsat-8 and Sentinel-2 Translation')
    
    parser.add_argument('--tensor_dir', type=str, required=True,
                        help='Directory containing tensor and angle files')
    parser.add_argument('--landsat_rsr_path', type=str, required=True,
                        help='Path to the Landsat-8 RSR file')
    parser.add_argument('--sentinel_rsr_path', type=str, required=True,
                        help='Path to the Sentinel-2 RSR file')
    parser.add_argument('--output_dir', type=str, default='./cyclegan_output',
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Beta1 parameter for Adam optimizer')
    parser.add_argument('--lambda_cycle', type=float, default=10.0,
                        help='Weight for cycle consistency loss')
    parser.add_argument('--lambda_identity', type=float, default=0.5,
                        help='Weight for identity loss')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Epoch interval to save model checkpoints')
    
    args = parser.parse_args()
    
    # Train the model
    train_cyclegan(
        tensor_dir=args.tensor_dir,
        landsat_rsr_path=args.landsat_rsr_path,
        sentinel_rsr_path=args.sentinel_rsr_path,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        save_interval=args.save_interval
    )