import os
import numpy as np
import rasterio
from rasterio.plot import show
import shutil
from tqdm import tqdm

def preprocess_satellite_data(base_dir):
    """
    Preprocess Landsat 8 and Sentinel-2 data applying appropriate scaling factors.
    Handles individual band files.
    
    Landsat 8: Scale factor 0.0000275, Offset -0.2
    Sentinel-2: Scale factor 0.0001, No offset
    
    Also copies metadata.json and angles.json to the scaled directory.
    
    Args:
        base_dir (str): Base directory containing the satellite data
    """
    # Stats counters
    landsat_processed = 0
    sentinel_processed = 0
    total_sites = 0
    total_dates = 0
    
    print(f"Starting preprocessing of satellite data from {base_dir}")
    
    # Get all site folders
    site_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and f != 'dataset_info.txt']
    total_sites = len(site_folders)
    print(f"Found {total_sites} site folders to process")
    
    for site in tqdm(site_folders, desc="Processing sites"):
        site_path = os.path.join(base_dir, site)
        print(f"\n{'='*50}")
        print(f"Processing site: {site}")
        
        # Get all date folders for the site
        date_folders = [d for d in os.listdir(site_path) if os.path.isdir(os.path.join(site_path, d))]
        site_dates = len([d for d in date_folders if not d.endswith('_scaled')])
        total_dates += site_dates
        print(f"Found {site_dates} date folders to process for site {site}")
        
        for date in tqdm(date_folders, desc=f"Dates in {site}", leave=False):
            # Skip if it's already a scaled directory
            if date.endswith('_scaled'):
                continue
                
            original_date_path = os.path.join(site_path, date)
            scaled_date_path = os.path.join(site_path, f"{date}_scaled")
            
            print(f"\n{'-'*50}")
            print(f"Processing date: {date} for site {site}")
            
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
                print(f"Found {len(landsat_files)} Landsat 8 files to process")
                
                for landsat_file in landsat_files:
                    src_file = os.path.join(landsat_dir, landsat_file)
                    dst_file = os.path.join(scaled_landsat_dir, landsat_file)
                    
                    # Process all files, even if they already exist
                    if os.path.exists(dst_file):
                        print(f"  Reprocessing {landsat_file} (overwriting existing file)")
                        # Continue with processing instead of skipping
                    
                    print(f"  Processing Landsat file: {landsat_file}")
                    
                    # Read, scale, and save the Landsat data
                    with rasterio.open(src_file) as src:
                        # Read the data
                        landsat_data = src.read()
                        
                        # Calculate and print stats before scaling
                        min_val = np.min(landsat_data)
                        max_val = np.max(landsat_data)
                        mean_val = np.mean(landsat_data)
                        std_val = np.std(landsat_data)
                        
                        print(f"    BEFORE SCALING - {landsat_file}:")
                        print(f"    Min: {min_val:.6f}, Max: {max_val:.6f}")
                        print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                        
                        # Apply scaling: scale factor 0.0000275, offset -0.2
                        landsat_scaled = landsat_data * 0.0000275 - 0.2
                        
                        # Calculate and print stats after scaling
                        min_scaled = np.min(landsat_scaled)
                        max_scaled = np.max(landsat_scaled)
                        mean_scaled = np.mean(landsat_scaled)
                        std_scaled = np.std(landsat_scaled)
                        
                        print(f"    AFTER SCALING - {landsat_file}:")
                        print(f"    Min: {min_scaled:.6f}, Max: {max_scaled:.6f}")
                        print(f"    Mean: {mean_scaled:.6f}, Std: {std_scaled:.6f}")
                        
                        # Create a new GeoTIFF with the same metadata
                        meta = src.meta.copy()
                        
                        with rasterio.open(dst_file, 'w', **meta) as dst:
                            dst.write(landsat_scaled)
                            
                        landsat_processed += 1
                        print(f"    Saved scaled Landsat file to {dst_file}")
            else:
                print(f"No Landsat 8 directory found for {site}/{date}")
            
            # Process Sentinel-2 data
            sentinel_dir = os.path.join(original_date_path, 'sentinel2')
            if os.path.exists(sentinel_dir):
                sentinel_files = [f for f in os.listdir(sentinel_dir) if f.endswith('.tif')]
                print(f"Found {len(sentinel_files)} Sentinel-2 files to process")
                
                for sentinel_file in sentinel_files:
                    src_file = os.path.join(sentinel_dir, sentinel_file)
                    dst_file = os.path.join(scaled_sentinel_dir, sentinel_file)
                    
                    # Process all files, even if they already exist
                    if os.path.exists(dst_file):
                        print(f"  Reprocessing {sentinel_file} (overwriting existing file)")
                        # Continue with processing instead of skipping
                    
                    print(f"  Processing Sentinel file: {sentinel_file}")
                    
                    # Read, scale, and save the Sentinel data
                    with rasterio.open(src_file) as src:
                        # Read the data
                        sentinel_data = src.read()
                        
                        # Calculate and print stats before scaling
                        min_val = np.min(sentinel_data)
                        max_val = np.max(sentinel_data)
                        mean_val = np.mean(sentinel_data)
                        std_val = np.std(sentinel_data)
                        
                        print(f"    BEFORE SCALING - {sentinel_file}:")
                        print(f"    Min: {min_val:.6f}, Max: {max_val:.6f}")
                        print(f"    Mean: {mean_val:.6f}, Std: {std_val:.6f}")
                        
                        # Apply scaling: scale factor 0.0001, no offset
                        sentinel_scaled = sentinel_data * 0.0001
                        
                        # Calculate and print stats after scaling
                        min_scaled = np.min(sentinel_scaled)
                        max_scaled = np.max(sentinel_scaled)
                        mean_scaled = np.mean(sentinel_scaled)
                        std_scaled = np.std(sentinel_scaled)
                        
                        print(f"    AFTER SCALING - {sentinel_file}:")
                        print(f"    Min: {min_scaled:.6f}, Max: {max_scaled:.6f}")
                        print(f"    Mean: {mean_scaled:.6f}, Std: {std_scaled:.6f}")
                        
                        # Create a new GeoTIFF with the same metadata
                        meta = src.meta.copy()
                        
                        with rasterio.open(dst_file, 'w', **meta) as dst:
                            dst.write(sentinel_scaled)
                            
                        sentinel_processed += 1
                        print(f"    Saved scaled Sentinel file to {dst_file}")
            else:
                print(f"No Sentinel-2 directory found for {site}/{date}")
            
            # Copy metadata.json to the scaled directory
            src_metadata = os.path.join(original_date_path, 'metadata.json')
            dst_metadata = os.path.join(scaled_date_path, 'metadata.json')
            if os.path.exists(src_metadata):
                shutil.copy2(src_metadata, dst_metadata)
                print(f"Copied metadata.json for {site}/{date}")
            
            # Copy angles.json to the scaled directory
            src_angles = os.path.join(original_date_path, 'angles.json')
            dst_angles = os.path.join(scaled_date_path, 'angles.json')
            if os.path.exists(src_angles):
                shutil.copy2(src_angles, dst_angles)
                print(f"Copied angles.json for {site}/{date}")

    # Print final stats summary
    print("\n" + "="*60)
    print("PREPROCESSING SUMMARY:")
    print("="*60)
    print(f"Total sites processed: {total_sites}")
    print(f"Total date folders processed: {total_dates}")
    print(f"Total Landsat 8 files processed: {landsat_processed}")
    print(f"Total Sentinel-2 files processed: {sentinel_processed}")
    print(f"Total files processed: {landsat_processed + sentinel_processed}")
    print("="*60)

if __name__ == "__main__":
    base_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data'
    preprocess_satellite_data(base_dir)
    print("Preprocessing complete. Scaled data is stored in [date]_scaled directories.")