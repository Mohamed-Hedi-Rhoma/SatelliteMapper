import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import pandas as pd

def visualize_satellite_pair(site_name, acquisition_date, use_scaled=True):
    """
    Visualize Landsat 8 and Sentinel-2 imagery for a specific site and date.
    
    Args:
        site_name (str): Name of the site (folder name)
        acquisition_date (str): Acquisition date in 'YYYY-MM-DD' format
        use_scaled (bool): Whether to use the scaled data (True) or original data (False)
    """
    # Define paths
    base_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data'
    
    if use_scaled:
        date_dir = os.path.join(base_dir, site_name, f"{acquisition_date}_scaled")
    else:
        date_dir = os.path.join(base_dir, site_name, acquisition_date)
        
    landsat_dir = os.path.join(date_dir, 'landsat8')
    sentinel_dir = os.path.join(date_dir, 'sentinel2')
    
    # Find image files
    landsat_files = [f for f in os.listdir(landsat_dir) if f.endswith('.tif')]
    sentinel_files = [f for f in os.listdir(sentinel_dir) if f.endswith('.tif')]
    
    if not landsat_files or not sentinel_files:
        print(f"No image pairs found for {site_name} on {acquisition_date}")
        return
    
    landsat_file = os.path.join(landsat_dir, landsat_files[0])
    sentinel_file = os.path.join(sentinel_dir, sentinel_files[0])
    
    # Read images
    with rasterio.open(landsat_file) as landsat_src, rasterio.open(sentinel_file) as sentinel_src:
        print(f"\n--- Image Pair Information ---")
        print(f"Site: {site_name}")
        print(f"Date: {acquisition_date}")
        print(f"Using {'scaled' if use_scaled else 'original'} data")
        
        # Read all bands
        landsat_img = landsat_src.read()
        sentinel_img = sentinel_src.read()
        
        # Get band information
        landsat_bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
        sentinel_bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
        
        # Print statistics
        print("\n--- Landsat 8 Band Statistics ---")
        landsat_stats = []
        for i, band_name in enumerate(landsat_bands):
            band_data = landsat_img[i]
            stats = {
                'Band': band_name,
                'Min': np.min(band_data),
                'Max': np.max(band_data),
                'Mean': np.mean(band_data),
                'Std': np.std(band_data)
            }
            landsat_stats.append(stats)
        
        # Display Landsat stats as a table
        landsat_df = pd.DataFrame(landsat_stats)
        print(landsat_df.to_string(index=False))
        
        print("\n--- Sentinel-2 Band Statistics ---")
        sentinel_stats = []
        for i, band_name in enumerate(sentinel_bands):
            band_data = sentinel_img[i]
            stats = {
                'Band': band_name,
                'Min': np.min(band_data),
                'Max': np.max(band_data),
                'Mean': np.mean(band_data),
                'Std': np.std(band_data)
            }
            sentinel_stats.append(stats)
        
        # Display Sentinel stats as a table
        sentinel_df = pd.DataFrame(sentinel_stats)
        print(sentinel_df.to_string(index=False))
        
        # Visualize the images
        fig, axs = plt.subplots(2, 4, figsize=(20, 10))
        
        # For scaled data, we can display directly without further scaling
        # For original data, we need to apply the scaling for visualization
        
        if use_scaled:
            # Landsat individual bands (already scaled)
            for i in range(3):  # Show only RGB bands individually
                band_data = landsat_img[i]
                vmin, vmax = np.percentile(band_data, (2, 98))  # Enhance contrast
                axs[0, i].imshow(band_data, cmap='gray', vmin=vmin, vmax=vmax)
                axs[0, i].set_title(f'Landsat 8 - {landsat_bands[i]}')
                axs[0, i].axis('off')
            
            # Landsat RGB (already scaled)
            rgb_landsat = np.dstack((landsat_img[2], landsat_img[1], landsat_img[0]))  # Red, Green, Blue
            # Clip to 0-1 range for visualization
            rgb_landsat = np.clip(rgb_landsat, 0, 1)
            axs[0, 3].imshow(rgb_landsat)
            axs[0, 3].set_title('Landsat 8 - RGB')
            axs[0, 3].axis('off')
            
            # Sentinel individual bands (already scaled)
            for i in range(3):  # Show only RGB bands individually
                band_data = sentinel_img[i]
                vmin, vmax = np.percentile(band_data, (2, 98))  # Enhance contrast
                axs[1, i].imshow(band_data, cmap='gray', vmin=vmin, vmax=vmax)
                axs[1, i].set_title(f'Sentinel-2 - {sentinel_bands[i]}')
                axs[1, i].axis('off')
            
            # Sentinel RGB (already scaled)
            rgb_sentinel = np.dstack((sentinel_img[2], sentinel_img[1], sentinel_img[0]))  # Red, Green, Blue
            # Clip to 0-1 range for visualization
            rgb_sentinel = np.clip(rgb_sentinel, 0, 1)
            axs[1, 3].imshow(rgb_sentinel)
            axs[1, 3].set_title('Sentinel-2 - RGB')
            axs[1, 3].axis('off')
        else:
            # Landsat individual bands (needs scaling)
            for i in range(3):  # Show only RGB bands individually
                band_data = landsat_img[i]
                vmin, vmax = np.percentile(band_data, (2, 98))  # Enhance contrast
                axs[0, i].imshow(band_data, cmap='gray', vmin=vmin, vmax=vmax)
                axs[0, i].set_title(f'Landsat 8 - {landsat_bands[i]}')
                axs[0, i].axis('off')
            
            # Landsat RGB (needs scaling)
            rgb_landsat = np.dstack((landsat_img[2], landsat_img[1], landsat_img[0]))  # Red, Green, Blue
            rgb_landsat = np.clip(rgb_landsat/10000, 0, 1)  # Scale for visualization
            axs[0, 3].imshow(rgb_landsat)
            axs[0, 3].set_title('Landsat 8 - RGB')
            axs[0, 3].axis('off')
            
            # Sentinel individual bands (needs scaling)
            for i in range(3):  # Show only RGB bands individually
                band_data = sentinel_img[i]
                vmin, vmax = np.percentile(band_data, (2, 98))  # Enhance contrast
                axs[1, i].imshow(band_data, cmap='gray', vmin=vmin, vmax=vmax)
                axs[1, i].set_title(f'Sentinel-2 - {sentinel_bands[i]}')
                axs[1, i].axis('off')
            
            # Sentinel RGB (needs scaling)
            rgb_sentinel = np.dstack((sentinel_img[2], sentinel_img[1], sentinel_img[0]))  # Red, Green, Blue
            rgb_sentinel = np.clip(rgb_sentinel/10000, 0, 1)  # Scale for visualization
            axs[1, 3].imshow(rgb_sentinel)
            axs[1, 3].set_title('Sentinel-2 - RGB')
            axs[1, 3].axis('off')
        
        plt.tight_layout()
        output_filename = f'{site_name}_{acquisition_date}_{"scaled" if use_scaled else "original"}_visualization.png'
        plt.savefig(os.path.join(date_dir, output_filename), dpi=300)
        plt.show()
        
        print(f"Visualization saved to {os.path.join(date_dir, output_filename)}")

if __name__ == "__main__":
    # Example usage:
    site_name = input("Enter site name: ")
    acquisition_date = input("Enter acquisition date (YYYY-MM-DD): ")
    
    use_scaled = input("Use scaled data? (y/n): ").lower() == 'y'
    
    visualize_satellite_pair(site_name, acquisition_date, use_scaled)