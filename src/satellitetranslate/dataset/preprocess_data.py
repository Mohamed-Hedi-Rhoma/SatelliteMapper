import os
import glob
import json
import shutil
import numpy as np
import rasterio
from rasterio.transform import Affine
import re
from tqdm import tqdm
import concurrent.futures

# Define directories
DATA_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data"
OUTPUT_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data_prepared"

# Define scaling factors and offsets
LANDSAT_SCALE_FACTOR = 0.0000275
LANDSAT_OFFSET = -0.2
SENTINEL_SCALE_FACTOR = 0.0001
SENTINEL_OFFSET = 0  # No offset for Sentinel-2

# Define the bands to process
LANDSAT_BANDS = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir1.tif', 'swir2.tif']
SENTINEL_BANDS = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir1.tif', 'swir2.tif']

def preprocess_landsat_image(input_path, output_path):
    """
    Preprocess a Landsat 8.
    Apply scale factor and offset: (Raw value * scale_factor) + offset
    
    Args:
        input_path (str): Path to input image file
        output_path (str): Path to output image file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read input image
    with rasterio.open(input_path) as src:
        # Read data
        data = src.read(1).astype(np.float32)
        
        # Apply scale factor and offset: (raw * scale_factor) + offset
        data = (data * LANDSAT_SCALE_FACTOR) + LANDSAT_OFFSET
        
        # Clip data to valid reflectance range [0, 1]
        data = np.clip(data, 0, 1)
        
        # Create output image with same metadata
        profile = src.profile.copy()
        profile.update({
            'dtype': 'float32',
            'driver': 'GTiff',
            'compress': 'lzw',
        })
        
        # Write output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)

def preprocess_sentinel_image(input_path, output_path):
    """
    Preprocess a Sentinel-2 image.
    Apply scale factor: Raw value * scale_factor (no offset)
    
    Args:
        input_path (str): Path to input image file
        output_path (str): Path to output image file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Read input image
    with rasterio.open(input_path) as src:
        # Read data
        data = src.read(1).astype(np.float32)
        
        # Apply scale factor: raw * scale_factor (no offset for Sentinel-2)
        data = data * SENTINEL_SCALE_FACTOR
        
        # Clip data to valid reflectance range [0, 1]
        data = np.clip(data, 0, 1)
        
        # Create output image with same metadata
        profile = src.profile.copy()
        profile.update({
            'dtype': 'float32',
            'driver': 'GTiff',
            'compress': 'lzw',
        })
        
        # Write output
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)

def process_image_pair(site_path):
    """
    Process all image pairs in a site directory.
    
    Args:
        site_path (str): Path to site directory
    """
    site_name = os.path.basename(site_path)
    print(f"Processing site: {site_name}")
    
    # Get all acquisition date directories
    acquisition_dirs = [d for d in glob.glob(os.path.join(site_path, "*")) 
                        if os.path.isdir(d) and not d.endswith("__pycache__")]
    
    for acq_dir in acquisition_dirs:
        acq_date = os.path.basename(acq_dir)
        print(f"  Processing acquisition date: {acq_date}")
        
        # Define paths
        sentinel_dir = os.path.join(acq_dir, "sentinel2")
        landsat_dir = os.path.join(acq_dir, "landsat8")
        
        # Define output directories
        output_site_dir = os.path.join(OUTPUT_DIR, site_name)
        output_acq_dir = os.path.join(output_site_dir, acq_date)
        output_sentinel_dir = os.path.join(output_acq_dir, "sentinel2")
        output_landsat_dir = os.path.join(output_acq_dir, "landsat8")
        
        # Create output directories
        os.makedirs(output_sentinel_dir, exist_ok=True)
        os.makedirs(output_landsat_dir, exist_ok=True)
        
        # Process Sentinel-2 images
        if os.path.exists(sentinel_dir):
            # Copy angles.json if it exists
            angles_json = os.path.join(sentinel_dir, "angles.json")
            if os.path.exists(angles_json):
                shutil.copy2(angles_json, os.path.join(output_sentinel_dir, "angles.json"))
            
            # Process each Sentinel-2 band
            for band in SENTINEL_BANDS:
                input_path = os.path.join(sentinel_dir, band)
                if os.path.exists(input_path):
                    output_path = os.path.join(output_sentinel_dir, band)
                    preprocess_sentinel_image(input_path, output_path)
                    print(f"    Processed Sentinel-2 {band}")
                else:
                    print(f"    Warning: Sentinel-2 {band} not found")
        else:
            print(f"  Warning: Sentinel-2 directory not found for {acq_date}")
        
        # Process Landsat 8 images
        if os.path.exists(landsat_dir):
            # Copy angles.json if it exists
            angles_json = os.path.join(landsat_dir, "angles.json")
            if os.path.exists(angles_json):
                shutil.copy2(angles_json, os.path.join(output_landsat_dir, "angles.json"))
            
            # Process each Landsat 8 band
            for band in LANDSAT_BANDS:
                input_path = os.path.join(landsat_dir, band)
                if os.path.exists(input_path):
                    output_path = os.path.join(output_landsat_dir, band)
                    preprocess_landsat_image(input_path, output_path)
                    print(f"    Processed Landsat 8 {band}")
                else:
                    print(f"    Warning: Landsat 8 {band} not found")
        else:
            print(f"  Warning: Landsat 8 directory not found for {acq_date}")

def process_all_sites_parallel():
    """
    Process all sites in parallel.
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all site directories
    site_dirs = [d for d in glob.glob(os.path.join(DATA_DIR, "*")) 
                if os.path.isdir(d) and not d.endswith("__pycache__")]
    
    print(f"Found {len(site_dirs)} sites to process")
    
    # Process each site in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(process_image_pair, site_dirs)
    
    print("All sites processed successfully")

def process_all_sites_sequential():
    """
    Process all sites sequentially (useful for debugging).
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get all site directories
    site_dirs = [d for d in glob.glob(os.path.join(DATA_DIR, "*")) 
                if os.path.isdir(d) and not d.endswith("__pycache__")]
    
    print(f"Found {len(site_dirs)} sites to process")
    
    # Process each site
    for site_dir in site_dirs:
        process_image_pair(site_dir)
    
    print("All sites processed successfully")

if __name__ == "__main__":
    # Uncomment one of these based on your preference
    # Process all sites in parallel (faster but harder to debug)
    process_all_sites_parallel()
    
    # Or process all sites sequentially (slower but easier to debug)
    # process_all_sites_sequential()