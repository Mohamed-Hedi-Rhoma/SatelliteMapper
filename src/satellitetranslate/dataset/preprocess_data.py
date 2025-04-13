import os
import numpy as np
import rasterio
from rasterio.plot import show
import shutil
from tqdm import tqdm

def preprocess_satellite_data(base_dir):
    """
    Preprocess Landsat 8 and Sentinel-2 data applying appropriate scaling factors.
    
    Landsat 8: Scale factor 0.0000275, Offset -0.2
    Sentinel-2: Scale factor 0.0001, No offset
    
    Args:
        base_dir (str): Base directory containing the satellite data
    """
    # Get all site folders
    site_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f != 'dataset_info.txt']
    
    for site in tqdm(site_folders, desc="Processing sites"):
        site_path = os.path.join(base_dir, site)
        
        # Get all date folders for the site
        date_folders = [d for d in os.listdir(site_path) if os.path.isdir(os.path.join(site_path, d))]
        
        for date in tqdm(date_folders, desc=f"Dates in {site}", leave=False):
            # Skip if it's already a scaled directory
            if date.endswith('_scaled'):
                continue
                
            original_date_path = os.path.join(site_path, date)
            scaled_date_path = os.path.join(site_path, f"{date}_scaled")
            
            # Create scaled directory
            os.makedirs(scaled_date_path, exist_ok=True)
            
            # Create landsat and sentinel subdirectories in the scaled directory
            scaled_landsat_dir = os.path.join(scaled_date_path, 'landsat8')
            scaled_sentinel_dir = os.path.join(scaled_date_path, 'sentinel2')
            os.makedirs(scaled_landsat_dir, exist_ok=True)
            os.makedirs(scaled_sentinel_dir, exist_ok=True)
            
            # Process Landsat 8 data
            landsat_dir = os.path.join(original_date_path, 'landsat8')
            if os.path.exists(landsat_dir):
                landsat_files = [f for f in os.listdir(landsat_dir) if f.endswith('.tif')]
                
                for landsat_file in landsat_files:
                    src_file = os.path.join(landsat_dir, landsat_file)
                    dst_file = os.path.join(scaled_landsat_dir, landsat_file)
                    
                    # Skip if already processed
                    if os.path.exists(dst_file):
                        continue
                    
                    # Read, scale, and save the Landsat data
                    with rasterio.open(src_file) as src:
                        # Read the data
                        landsat_data = src.read()
                        
                        # Apply scaling: scale factor 0.0000275, offset -0.2
                        landsat_scaled = landsat_data * 0.0000275 - 0.2
                        
                        # Create a new GeoTIFF with the same metadata
                        meta = src.meta.copy()
                        
                        with rasterio.open(dst_file, 'w', **meta) as dst:
                            dst.write(landsat_scaled)
            
            # Process Sentinel-2 data
            sentinel_dir = os.path.join(original_date_path, 'sentinel2')
            if os.path.exists(sentinel_dir):
                sentinel_files = [f for f in os.listdir(sentinel_dir) if f.endswith('.tif')]
                
                for sentinel_file in sentinel_files:
                    src_file = os.path.join(sentinel_dir, sentinel_file)
                    dst_file = os.path.join(scaled_sentinel_dir, sentinel_file)
                    
                    # Skip if already processed
                    if os.path.exists(dst_file):
                        continue
                    
                    # Read, scale, and save the Sentinel data
                    with rasterio.open(src_file) as src:
                        # Read the data
                        sentinel_data = src.read()
                        
                        # Apply scaling: scale factor 0.0001, no offset
                        sentinel_scaled = sentinel_data * 0.0001
                        
                        # Create a new GeoTIFF with the same metadata
                        meta = src.meta.copy()
                        
                        with rasterio.open(dst_file, 'w', **meta) as dst:
                            dst.write(sentinel_scaled)
            
            # Copy metadata.json to the scaled directory
            src_metadata = os.path.join(original_date_path, 'metadata.json')
            dst_metadata = os.path.join(scaled_date_path, 'metadata.json')
            if os.path.exists(src_metadata) and not os.path.exists(dst_metadata):
                shutil.copy2(src_metadata, dst_metadata)

if __name__ == "__main__":
    base_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data'
    preprocess_satellite_data(base_dir)
    print("Preprocessing complete. Scaled data is stored in [date]_scaled directories.")