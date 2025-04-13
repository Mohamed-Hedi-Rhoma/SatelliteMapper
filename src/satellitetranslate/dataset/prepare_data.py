import os
import numpy as np
import torch
import rasterio
from tqdm import tqdm
from pathlib import Path
import random
import json
import copy

def prepare_satellite_dataset(base_dir, output_dir, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Prepare satellite dataset for PyTorch by creating tensor files.
    
    Args:
        base_dir (str): Base directory containing the scaled satellite data
        output_dir (str): Output directory for the tensor files
        val_ratio (float): Ratio of validation data (default: 0.1)
        test_ratio (float): Ratio of test data (default: 0.1)
        seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # List of band names in the order we want them
    landsat_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    sentinel_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    
    # Get all valid data pairs
    print("Finding all scaled image pairs...")
    pairs = []
    
    # Get all site folders
    site_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
    
    for site in tqdm(site_folders, desc="Processing sites"):
        site_path = os.path.join(base_dir, site)
        
        # Get all scaled date folders for the site
        date_folders = [d for d in os.listdir(site_path) if os.path.isdir(os.path.join(site_path, d)) and d.endswith('_scaled')]
        
        for date in date_folders:
            date_path = os.path.join(site_path, date)
            landsat_dir = os.path.join(date_path, 'landsat8')
            sentinel_dir = os.path.join(date_path, 'sentinel2')
            
            # Check if both landsat and sentinel data exist
            if not (os.path.exists(landsat_dir) and os.path.exists(sentinel_dir)):
                continue
            
            # Check if all bands exist for both sensors
            landsat_files = {}
            sentinel_files = {}
            
            # Find all band files
            for file in os.listdir(landsat_dir):
                if file.endswith('.tif'):
                    # Extract band name from filename (e.g., landsat8_site_date_blue.tif)
                    for band in landsat_bands:
                        if f"_{band}.tif" in file:
                            landsat_files[band] = os.path.join(landsat_dir, file)
            
            for file in os.listdir(sentinel_dir):
                if file.endswith('.tif'):
                    for band in sentinel_bands:
                        if f"_{band}.tif" in file:
                            sentinel_files[band] = os.path.join(sentinel_dir, file)
            
            # Check if all bands are present and if angles.json exists
            angles_file = os.path.join(date_path, 'angles.json')
            
            if (len(landsat_files) == len(landsat_bands) and 
                len(sentinel_files) == len(sentinel_bands) and
                os.path.exists(angles_file)):
                
                # Read the angles.json file
                with open(angles_file, 'r') as f:
                    angles_data = json.load(f)
                
                pairs.append({
                    'site': site,
                    'date': date,
                    'landsat_files': landsat_files,
                    'sentinel_files': sentinel_files,
                    'path': date_path,
                    'angles': angles_data
                })
    
    print(f"Found {len(pairs)} valid image pairs")
    
    # Shuffle the pairs
    random.shuffle(pairs)
    
    # Split into train, validation, and test sets
    num_samples = len(pairs)
    num_test = int(num_samples * test_ratio)
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_test - num_val
    
    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:num_train+num_val]
    test_pairs = pairs[num_train+num_val:]
    
    print(f"Split into {len(train_pairs)} training, {len(val_pairs)} validation, and {len(test_pairs)} test pairs")
    
    # Create a log file with the dataset split information
    with open(os.path.join(output_dir, 'dataset_split_info.json'), 'w') as f:
        json.dump({
            'train_pairs': [{'site': p['site'], 'date': p['date']} for p in train_pairs],
            'val_pairs': [{'site': p['site'], 'date': p['date']} for p in val_pairs],
            'test_pairs': [{'site': p['site'], 'date': p['date']} for p in test_pairs],
            'total_samples': num_samples,
            'train_samples': num_train,
            'val_samples': num_val,
            'test_samples': num_test,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio
        }, f, indent=2)
    
    # Process each split
    for split_name, split_pairs in [('train', train_pairs), ('valid', val_pairs), ('test', test_pairs)]:
        # Skip if the files already exist
        if (os.path.exists(os.path.join(output_dir, f'{split_name}_x.pth')) and 
            os.path.exists(os.path.join(output_dir, f'{split_name}_y.pth')) and
            os.path.exists(os.path.join(output_dir, f'angles_{split_name}_x.json')) and
            os.path.exists(os.path.join(output_dir, f'angles_{split_name}_y.json'))):
            print(f"{split_name} files already exist, skipping...")
            continue
        
        num_pairs = len(split_pairs)
        
        # Pre-allocate tensors for Landsat (x) and Sentinel (y) data
        x_data = torch.zeros((num_pairs, 6, 128, 128), dtype=torch.float32)
        y_data = torch.zeros((num_pairs, 6, 384, 384), dtype=torch.float32)
        
        # Create lists to store angle information
        angles_x = []  # For Landsat
        angles_y = []  # For Sentinel
        
        print(f"Processing {split_name} set ({num_pairs} pairs)...")
        
        for i, pair in enumerate(tqdm(split_pairs, desc=f"Loading {split_name} data")):
            # Load Landsat bands in the correct order
            for band_idx, band_name in enumerate(landsat_bands):
                with rasterio.open(pair['landsat_files'][band_name]) as src:
                    landsat_band = src.read(1)  # Read the first band
                    # Ensure it's 128x128, crop or pad if necessary
                    if landsat_band.shape != (128, 128):
                        if landsat_band.shape[0] > 128 and landsat_band.shape[1] > 128:
                            # Crop to center 128x128
                            h, w = landsat_band.shape
                            start_h = (h - 128) // 2
                            start_w = (w - 128) // 2
                            landsat_band = landsat_band[start_h:start_h+128, start_w:start_w+128]
                        else:
                            # Pad to 128x128
                            padded = np.zeros((128, 128), dtype=landsat_band.dtype)
                            h, w = min(landsat_band.shape[0], 128), min(landsat_band.shape[1], 128)
                            padded[:h, :w] = landsat_band[:h, :w]
                            landsat_band = padded
                    
                    # Convert to tensor and add to data array
                    x_data[i, band_idx] = torch.from_numpy(landsat_band.astype(np.float32))
            
            # Load Sentinel bands in the correct order
            for band_idx, band_name in enumerate(sentinel_bands):
                with rasterio.open(pair['sentinel_files'][band_name]) as src:
                    sentinel_band = src.read(1)  # Read the first band
                    # Ensure it's 384x384, crop or pad if necessary
                    if sentinel_band.shape != (384, 384):
                        if sentinel_band.shape[0] > 384 and sentinel_band.shape[1] > 384:
                            # Crop to center 384x384
                            h, w = sentinel_band.shape
                            start_h = (h - 384) // 2
                            start_w = (w - 384) // 2
                            sentinel_band = sentinel_band[start_h:start_h+384, start_w:start_w+384]
                        else:
                            # Pad to 384x384
                            padded = np.zeros((384, 384), dtype=sentinel_band.dtype)
                            h, w = min(sentinel_band.shape[0], 384), min(sentinel_band.shape[1], 384)
                            padded[:h, :w] = sentinel_band[:h, :w]
                            sentinel_band = padded
                    
                    # Convert to tensor and add to data array
                    y_data[i, band_idx] = torch.from_numpy(sentinel_band.astype(np.float32))
            
            # Add angle data
            angles_x.append({
                'site': pair['site'],
                'date': pair['date'],
                'landsat_date': pair['angles']['landsat']['date'],
                'angles': copy.deepcopy(pair['angles']['landsat']['angles'])
            })
            
            angles_y.append({
                'site': pair['site'],
                'date': pair['date'],
                'sentinel_date': pair['angles']['sentinel']['date'],
                'angles': copy.deepcopy(pair['angles']['sentinel']['angles'])
            })
        
        # Save the tensors
        torch.save(x_data, os.path.join(output_dir, f'{split_name}_x.pth'))
        torch.save(y_data, os.path.join(output_dir, f'{split_name}_y.pth'))
        
        # Save angle files (matching the ordering of the tensors)
        with open(os.path.join(output_dir, f'angles_{split_name}_x.json'), 'w') as f:
            json.dump(angles_x, f, indent=2)
            
        with open(os.path.join(output_dir, f'angles_{split_name}_y.json'), 'w') as f:
            json.dump(angles_y, f, indent=2)
        
        print(f"Saved {split_name} tensors with shapes: x={x_data.shape}, y={y_data.shape}")
        print(f"Saved corresponding angle files with {len(angles_x)} entries each")
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    base_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data'
    output_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/tensors'
    prepare_satellite_dataset(base_dir, output_dir)