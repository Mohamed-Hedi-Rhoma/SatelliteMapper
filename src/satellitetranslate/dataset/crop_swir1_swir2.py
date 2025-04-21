import os
import glob
import rasterio
from rasterio.windows import Window
import numpy as np

# Base directory for Sentinel-2 data
SENTINEL_DATA_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data_sentinel2"

# Define the bands to crop
SWIR_BANDS = ['swir1.tif', 'swir2.tif']

def crop_swir_bands(site_dir):
    """
    Crops SWIR1 and SWIR2 bands from 384x384 to 192x192 pixels centered on the same area.
    
    Args:
        site_dir (str): Path to the site directory containing date folders
    """
    print(f"Processing site: {os.path.basename(site_dir)}")
    
    # Get all date directories in the site folder
    date_dirs = [d for d in glob.glob(os.path.join(site_dir, "*")) 
                if os.path.isdir(d) and not d.endswith("__pycache__")]
    
    for date_dir in date_dirs:
        date = os.path.basename(date_dir)
        print(f"  Processing date: {date}")
        
        for band_name in SWIR_BANDS:
            band_path = os.path.join(date_dir, band_name)
            
            # Skip if file doesn't exist
            if not os.path.exists(band_path):
                print(f"    {band_name} not found for {date}. Skipping.")
                continue
            
            print(f"    Cropping {band_name}...")
            
            try:
                # Open the raster file
                with rasterio.open(band_path) as src:
                    # Get the current dimensions
                    height, width = src.height, src.width
                    
                    # Check if already at target size
                    if height == 192 and width == 192:
                        print(f"    {band_name} is already 192x192. Skipping.")
                        continue
                    
                    # Calculate the window to read (centered)
                    x_offset = (width - 192) // 2
                    y_offset = (height - 192) // 2
                    window = Window(x_offset, y_offset, 192, 192)
                    
                    # Read the data and metadata
                    data = src.read(1, window=window)
                    profile = src.profile.copy()
                    
                    # Update the profile for the new dimensions
                    profile.update({
                        'height': 192,
                        'width': 192,
                        'transform': rasterio.windows.transform(window, src.transform)
                    })
                    
                # Save the cropped data
                with rasterio.open(band_path, 'w', **profile) as dst:
                    dst.write(data, 1)
                
                print(f"    Successfully cropped {band_name} to 192x192 pixels")
                
            except Exception as e:
                print(f"    Error processing {band_name}: {e}")

def main():
    """
    Main function to process all sites.
    """
    print("Starting Sentinel-2 SWIR band cropping process...")
    
    # Get all site directories
    site_dirs = [d for d in glob.glob(os.path.join(SENTINEL_DATA_DIR, "*")) 
                if os.path.isdir(d) and not d.endswith("__pycache__")]
    
    print(f"Found {len(site_dirs)} site directories")
    
    # Process each site
    for site_dir in site_dirs:
        crop_swir_bands(site_dir)
    
    print("SWIR band cropping completed successfully.")

if __name__ == '__main__':
    main()