import os
import numpy as np
import torch
import rasterio
from tqdm import tqdm
from pathlib import Path
import random
import json
import copy
import time

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
    # Start time
    start_time = time.time()
    print(f"\n{'='*80}")
    print(f"STARTING DATASET PREPARATION")
    print(f"{'='*80}")
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Validation ratio: {val_ratio}, Test ratio: {test_ratio}, Seed: {seed}")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    print(f"Random seeds set to {seed}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/confirmed: {output_dir}")
    
    # List of band names in the order we want them
    landsat_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    sentinel_bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    print(f"Landsat bands (in order): {landsat_bands}")
    print(f"Sentinel bands (in order): {sentinel_bands}")
    
    # Get all valid data pairs
    print(f"\n{'-'*80}")
    print("Finding all scaled image pairs...")
    pairs = []
    
    # Site and date counters
    total_sites = 0
    total_dates = 0
    valid_pairs = 0
    invalid_pairs = 0
    missing_landsat = 0
    missing_sentinel = 0
    missing_bands = 0
    missing_angles = 0
    
    # Get all site folders
    site_folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
    total_sites = len(site_folders)
    print(f"Found {total_sites} site folders")
    
    for site in tqdm(site_folders, desc="Processing sites"):
        site_path = os.path.join(base_dir, site)
        site_dates = 0
        site_valid_pairs = 0
        
        print(f"\n{'-'*70}")
        print(f"Processing site: {site}")
        
        # Get all scaled date folders for the site
        date_folders = [d for d in os.listdir(site_path) if os.path.isdir(os.path.join(site_path, d)) and d.endswith('_scaled')]
        site_dates = len(date_folders)
        total_dates += site_dates
        print(f"Found {site_dates} scaled date folders for site {site}")
        
        for date in date_folders:
            date_path = os.path.join(site_path, date)
            landsat_dir = os.path.join(date_path, 'landsat8')
            sentinel_dir = os.path.join(date_path, 'sentinel2')
            
            print(f"  Checking pair: {site}/{date}")
            
            # Check if both landsat and sentinel data exist
            if not os.path.exists(landsat_dir):
                print(f"    Missing Landsat directory at {landsat_dir}")
                missing_landsat += 1
                invalid_pairs += 1
                continue
                
            if not os.path.exists(sentinel_dir):
                print(f"    Missing Sentinel directory at {sentinel_dir}")
                missing_sentinel += 1
                invalid_pairs += 1
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
            
            # Print found bands
            print(f"    Found Landsat bands: {list(landsat_files.keys())}")
            print(f"    Found Sentinel bands: {list(sentinel_files.keys())}")
            
            # Check if all bands are present and if angles.json exists
            angles_file = os.path.join(date_path, 'angles.json')
            
            if len(landsat_files) != len(landsat_bands) or len(sentinel_files) != len(sentinel_bands):
                print(f"    Missing bands: Landsat has {len(landsat_files)}/{len(landsat_bands)}, Sentinel has {len(sentinel_files)}/{len(sentinel_bands)}")
                missing_bands += 1
                invalid_pairs += 1
                continue
                
            if not os.path.exists(angles_file):
                print(f"    Missing angles.json file at {angles_file}")
                missing_angles += 1
                invalid_pairs += 1
                continue
            
            # Read the angles.json file
            with open(angles_file, 'r') as f:
                angles_data = json.load(f)
                print(f"    Successfully read angles.json")
            
            # Valid pair found
            pairs.append({
                'site': site,
                'date': date,
                'landsat_files': landsat_files,
                'sentinel_files': sentinel_files,
                'path': date_path,
                'angles': angles_data
            })
            
            valid_pairs += 1
            site_valid_pairs += 1
            print(f"    [OK] Valid pair found: {site}/{date}")
        
        print(f"  Site summary: {site_valid_pairs}/{site_dates} valid pairs for site {site}")
    
    print(f"\n{'-'*80}")
    print(f"DATASET PAIRING SUMMARY:")
    print(f"{'-'*80}")
    print(f"Total sites: {total_sites}")
    print(f"Total date folders: {total_dates}")
    print(f"Valid pairs: {valid_pairs}")
    print(f"Invalid pairs: {invalid_pairs}")
    print(f"  - Missing Landsat: {missing_landsat}")
    print(f"  - Missing Sentinel: {missing_sentinel}")
    print(f"  - Missing bands: {missing_bands}")
    print(f"  - Missing angles: {missing_angles}")
    
    if valid_pairs == 0:
        print("ERROR: No valid image pairs found. Exiting...")
        return
    
    # Shuffle the pairs
    random.shuffle(pairs)
    print(f"Shuffled {len(pairs)} valid image pairs")
    
    # Split into train, validation, and test sets
    num_samples = len(pairs)
    num_test = int(num_samples * test_ratio)
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_test - num_val
    
    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:num_train+num_val]
    test_pairs = pairs[num_train+num_val:]
    
    print(f"\n{'-'*80}")
    print(f"DATASET SPLITTING:")
    print(f"{'-'*80}")
    print(f"Total samples: {num_samples}")
    print(f"Training samples: {num_train} ({num_train/num_samples:.2%})")
    print(f"Validation samples: {num_val} ({num_val/num_samples:.2%})")
    print(f"Testing samples: {num_test} ({num_test/num_samples:.2%})")
    
    # Create a log file with the dataset split information
    split_info_path = os.path.join(output_dir, 'dataset_split_info.json')
    with open(split_info_path, 'w') as f:
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
    print(f"Dataset split information saved to {split_info_path}")
    
    # Stats tracking
    band_stats = {
        'landsat': {band: {'min': float('inf'), 'max': float('-inf'), 'mean': [], 'std': []} for band in landsat_bands},
        'sentinel': {band: {'min': float('inf'), 'max': float('-inf'), 'mean': [], 'std': []} for band in sentinel_bands}
    }
    
    # Process each split
    for split_name, split_pairs in [('train', train_pairs), ('valid', val_pairs), ('test', test_pairs)]:
        print(f"\n{'-'*80}")
        print(f"PROCESSING {split_name.upper()} SPLIT:")
        print(f"{'-'*80}")
        
        # Force reprocessing even if files exist
        if (os.path.exists(os.path.join(output_dir, f'{split_name}_x.pth')) and 
            os.path.exists(os.path.join(output_dir, f'{split_name}_y.pth')) and
            os.path.exists(os.path.join(output_dir, f'angles_{split_name}_x.json')) and
            os.path.exists(os.path.join(output_dir, f'angles_{split_name}_y.json'))):
            print(f"{split_name} files already exist, but will be reprocessed...")
        
        num_pairs = len(split_pairs)
        
        # Pre-allocate tensors for Landsat (x) and Sentinel (y) data
        x_data = torch.zeros((num_pairs, 6, 128, 128), dtype=torch.float32)
        y_data = torch.zeros((num_pairs, 6, 384, 384), dtype=torch.float32)
        
        # Create lists to store angle information
        angles_x = []  # For Landsat
        angles_y = []  # For Sentinel
        
        print(f"Processing {split_name} set with {num_pairs} pairs...")
        
        # Track resize operations
        crop_count = {'landsat': 0, 'sentinel': 0}
        pad_count = {'landsat': 0, 'sentinel': 0}
        
        for i, pair in enumerate(tqdm(split_pairs, desc=f"Loading {split_name} data")):
            print(f"\nPair {i+1}/{num_pairs}: {pair['site']}/{pair['date']}")
            
            # Load Landsat bands in the correct order
            for band_idx, band_name in enumerate(landsat_bands):
                with rasterio.open(pair['landsat_files'][band_name]) as src:
                    landsat_band = src.read(1)  # Read the first band
                    orig_shape = landsat_band.shape
                    
                    # Calculate stats
                    band_min = np.min(landsat_band)
                    band_max = np.max(landsat_band)
                    band_mean = np.mean(landsat_band)
                    band_std = np.std(landsat_band)
                    
                    # Update global stats
                    band_stats['landsat'][band_name]['min'] = min(band_stats['landsat'][band_name]['min'], band_min)
                    band_stats['landsat'][band_name]['max'] = max(band_stats['landsat'][band_name]['max'], band_max)
                    band_stats['landsat'][band_name]['mean'].append(band_mean)
                    band_stats['landsat'][band_name]['std'].append(band_std)
                    
                    print(f"  Landsat {band_name} - Shape: {orig_shape}, Min: {band_min:.6f}, Max: {band_max:.6f}, Mean: {band_mean:.6f}, Std: {band_std:.6f}")
                    
                    
                    # Convert to tensor and add to data array
                    x_data[i, band_idx] = torch.from_numpy(landsat_band.astype(np.float32))
            
            # Load Sentinel bands in the correct order
            for band_idx, band_name in enumerate(sentinel_bands):
                with rasterio.open(pair['sentinel_files'][band_name]) as src:
                    sentinel_band = src.read(1)  # Read the first band
                    orig_shape = sentinel_band.shape
                    
                    # Calculate stats
                    band_min = np.min(sentinel_band)
                    band_max = np.max(sentinel_band)
                    band_mean = np.mean(sentinel_band)
                    band_std = np.std(sentinel_band)
                    
                    # Update global stats
                    band_stats['sentinel'][band_name]['min'] = min(band_stats['sentinel'][band_name]['min'], band_min)
                    band_stats['sentinel'][band_name]['max'] = max(band_stats['sentinel'][band_name]['max'], band_max)
                    band_stats['sentinel'][band_name]['mean'].append(band_mean)
                    band_stats['sentinel'][band_name]['std'].append(band_std)
                    
                    print(f"  Sentinel {band_name} - Shape: {orig_shape}, Min: {band_min:.6f}, Max: {band_max:.6f}, Mean: {band_mean:.6f}, Std: {band_std:.6f}")
                    
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
            
            print(f"  Added angle data for both sensors")
        
        # Print resize operations
        print(f"\nResize operations for {split_name} set:")
        print(f"  Landsat: {crop_count['landsat']} crops, {pad_count['landsat']} pads")
        print(f"  Sentinel: {crop_count['sentinel']} crops, {pad_count['sentinel']} pads")
        
        # Print tensor stats
        print(f"\nTensor statistics for {split_name} set:")
        print(f"  x_data shape: {x_data.shape}, dtype: {x_data.dtype}")
        print(f"  y_data shape: {y_data.shape}, dtype: {y_data.dtype}")
        print(f"  x_data min: {x_data.min():.6f}, max: {x_data.max():.6f}, mean: {x_data.mean():.6f}, std: {x_data.std():.6f}")
        print(f"  y_data min: {y_data.min():.6f}, max: {y_data.max():.6f}, mean: {y_data.mean():.6f}, std: {y_data.std():.6f}")
        
        # Save the tensors
        x_path = os.path.join(output_dir, f'{split_name}_x.pth')
        y_path = os.path.join(output_dir, f'{split_name}_y.pth')
        torch.save(x_data, x_path)
        torch.save(y_data, y_path)
        print(f"Saved tensors to {x_path} and {y_path}")
        
        # Save angle files (matching the ordering of the tensors)
        angles_x_path = os.path.join(output_dir, f'angles_{split_name}_x.json')
        angles_y_path = os.path.join(output_dir, f'angles_{split_name}_y.json')
        with open(angles_x_path, 'w') as f:
            json.dump(angles_x, f, indent=2)
            
        with open(angles_y_path, 'w') as f:
            json.dump(angles_y, f, indent=2)
            
        print(f"Saved angle files to {angles_x_path} and {angles_y_path}")
    
    # Calculate and print global band statistics
    print(f"\n{'-'*80}")
    print(f"GLOBAL BAND STATISTICS:")
    print(f"{'-'*80}")
    
    # Landsat bands
    print("\nLandsat bands:")
    for band in landsat_bands:
        stats = band_stats['landsat'][band]
        mean_of_means = np.mean(stats['mean'])
        mean_of_stds = np.mean(stats['std'])
        print(f"  {band}:")
        print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
        print(f"    Mean: {mean_of_means:.6f}, Std: {mean_of_stds:.6f}")
    
    # Sentinel bands
    print("\nSentinel bands:")
    for band in sentinel_bands:
        stats = band_stats['sentinel'][band]
        mean_of_means = np.mean(stats['mean'])
        mean_of_stds = np.mean(stats['std'])
        print(f"  {band}:")
        print(f"    Min: {stats['min']:.6f}, Max: {stats['max']:.6f}")
        print(f"    Mean: {mean_of_means:.6f}, Std: {mean_of_stds:.6f}")
    
    # Save band statistics
    stats_path = os.path.join(output_dir, 'band_statistics.json')
    with open(stats_path, 'w') as f:
        # Convert numpy values to native Python types for JSON serialization
        for sensor in band_stats:
            for band in band_stats[sensor]:
                stats = band_stats[sensor][band]
                stats['mean'] = float(np.mean(stats['mean']))
                stats['std'] = float(np.mean(stats['std']))
                stats['min'] = float(stats['min'])
                stats['max'] = float(stats['max'])
        
        json.dump(band_stats, f, indent=2)
    print(f"Saved band statistics to {stats_path}")
    
    # Print elapsed time
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n{'-'*80}")
    print(f"DATASET PREPARATION COMPLETE!")
    print(f"{'-'*80}")
    print(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{seconds:.2f}")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    base_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data'
    output_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data_pth'
    prepare_satellite_dataset(base_dir, output_dir)