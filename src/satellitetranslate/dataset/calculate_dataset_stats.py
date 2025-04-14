import os
import torch
import json
import numpy as np
from tqdm import tqdm
import math

def calculate_statistics(tensor_dir, output_dir=None):
    """
    Calculate mean and standard deviation for each band of satellite imagery
    and for the angles (using cosine of angles) using the quantile method.
    With additional debugging information.
    
    Args:
        tensor_dir (str): Directory containing tensor and angle files
        output_dir (str, optional): Directory to save the statistics files.
                                   If None, uses tensor_dir.
    """
    if output_dir is None:
        output_dir = tensor_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading tensors...")
    
    # Load all tensor data
    train_x = torch.load(os.path.join(tensor_dir, 'train_x.pth'))
    train_y = torch.load(os.path.join(tensor_dir, 'train_y.pth'))
    valid_x = torch.load(os.path.join(tensor_dir, 'valid_x.pth'))
    valid_y = torch.load(os.path.join(tensor_dir, 'valid_y.pth'))
    test_x = torch.load(os.path.join(tensor_dir, 'test_x.pth'))
    test_y = torch.load(os.path.join(tensor_dir, 'test_y.pth'))
    
    # Combine all data
    all_landsat = torch.cat([train_x, valid_x, test_x], dim=0)
    all_sentinel = torch.cat([train_y, valid_y, test_y], dim=0)
    
    print(f"Combined dataset sizes: Landsat {all_landsat.shape}, Sentinel {all_sentinel.shape}")
    
    # Add debug info for data ranges
    print(f"Landsat data range: min={all_landsat.min().item()}, max={all_landsat.max().item()}")
    print(f"Sentinel data range: min={all_sentinel.min().item()}, max={all_sentinel.max().item()}")
    
    # Check for NaN or Infinity values
    print(f"Landsat NaN count: {torch.isnan(all_landsat).sum().item()}")
    print(f"Landsat Inf count: {torch.isinf(all_landsat).sum().item()}")
    print(f"Sentinel NaN count: {torch.isnan(all_sentinel).sum().item()}")
    print(f"Sentinel Inf count: {torch.isinf(all_sentinel).sum().item()}")
    
    # Load angle data
    with open(os.path.join(tensor_dir, 'angles_train_x.json'), 'r') as f:
        train_angles_x = json.load(f)
    with open(os.path.join(tensor_dir, 'angles_valid_x.json'), 'r') as f:
        valid_angles_x = json.load(f)
    with open(os.path.join(tensor_dir, 'angles_test_x.json'), 'r') as f:
        test_angles_x = json.load(f)
    
    with open(os.path.join(tensor_dir, 'angles_train_y.json'), 'r') as f:
        train_angles_y = json.load(f)
    with open(os.path.join(tensor_dir, 'angles_valid_y.json'), 'r') as f:
        valid_angles_y = json.load(f)
    with open(os.path.join(tensor_dir, 'angles_test_y.json'), 'r') as f:
        test_angles_y = json.load(f)
    
    # Combine all angle data
    all_angles_x = train_angles_x + valid_angles_x + test_angles_x
    all_angles_y = train_angles_y + valid_angles_y + test_angles_y
    
    print(f"Combined angle datasets: Landsat {len(all_angles_x)}, Sentinel {len(all_angles_y)}")
    
    # Convert angles to tensors and apply cosine
    angles_x_tensor = torch.zeros((len(all_angles_x), 4), dtype=torch.float32)
    angles_y_tensor = torch.zeros((len(all_angles_y), 4), dtype=torch.float32)
    
    for i, sample in enumerate(all_angles_x):
        angles = sample['angles']
        # Order: solar_azimuth, solar_zenith, view_azimuth, view_zenith
        angle_values = [
            angles.get('solar_azimuth', 0),
            angles.get('solar_zenith', 0),
            angles.get('view_azimuth', 0),
            angles.get('view_zenith', 0)
        ]
        # Replace None with 0
        angle_values = [0 if v is None else v for v in angle_values]
        # Convert to radians and apply cosine
        angle_values = [math.cos(math.radians(v)) if v is not None else 0 for v in angle_values]
        angles_x_tensor[i] = torch.tensor(angle_values, dtype=torch.float32)
    
    for i, sample in enumerate(all_angles_y):
        angles = sample['angles']
        angle_values = [
            angles.get('solar_azimuth', 0),
            angles.get('solar_zenith', 0),
            angles.get('view_azimuth', 0),
            angles.get('view_zenith', 0)
        ]
        angle_values = [0 if v is None else v for v in angle_values]
        # Convert to radians and apply cosine
        angle_values = [math.cos(math.radians(v)) if v is not None else 0 for v in angle_values]
        angles_y_tensor[i] = torch.tensor(angle_values, dtype=torch.float32)
    
    # Calculate quantile-based statistics for Landsat bands
    landsat_means = torch.zeros(6, dtype=torch.float32)
    landsat_stds = torch.zeros(6, dtype=torch.float32)
    
    print("Calculating Landsat statistics...")
    for band in tqdm(range(6)):
        band_data = all_landsat[:, band, :, :].flatten()
        
        # Debug info for each band
        print(f"\nLandsat Band {band} stats:")
        print(f"  Raw data shape: {band_data.shape}")
        print(f"  Min: {band_data.min().item()}, Max: {band_data.max().item()}")
        print(f"  Mean: {band_data.mean().item()}, Std: {band_data.std().item()}")
        
        # Sort the data
        sorted_data, _ = torch.sort(band_data)
        
        # Calculate quantiles (10% and 90%)
        q10_idx = int(0.1 * len(sorted_data))
        q90_idx = int(0.9 * len(sorted_data))
        q10 = sorted_data[q10_idx]
        q90 = sorted_data[q90_idx]
        
        print(f"  10% quantile: {q10.item()}, 90% quantile: {q90.item()}")
        
        # Use data between q10 and q90 for statistics
        filtered_data = band_data[(band_data >= q10) & (band_data <= q90)]
        
        print(f"  Filtered data shape: {filtered_data.shape}")
        print(f"  Filtered Min: {filtered_data.min().item()}, Max: {filtered_data.max().item()}")
        print(f"  Filtered Mean: {filtered_data.mean().item()}, Std: {filtered_data.std().item()}")
        
        landsat_means[band] = torch.mean(filtered_data)
        landsat_stds[band] = torch.std(filtered_data)
    
    # Calculate quantile-based statistics for Sentinel bands
    sentinel_means = torch.zeros(6, dtype=torch.float32)
    sentinel_stds = torch.zeros(6, dtype=torch.float32)
    
    print("\nCalculating Sentinel statistics...")
    for band in tqdm(range(6)):
        band_data = all_sentinel[:, band, :, :].flatten()
        
        # Debug info for each band
        print(f"\nSentinel Band {band} stats:")
        print(f"  Raw data shape: {band_data.shape}")
        print(f"  Min: {band_data.min().item()}, Max: {band_data.max().item()}")
        print(f"  Mean: {band_data.mean().item()}, Std: {band_data.std().item()}")
        
        # Sort the data
        sorted_data, _ = torch.sort(band_data)
        
        # Calculate quantiles (10% and 90%)
        q10_idx = int(0.1 * len(sorted_data))
        q90_idx = int(0.9 * len(sorted_data))
        q10 = sorted_data[q10_idx]
        q90 = sorted_data[q90_idx]
        
        print(f"  10% quantile: {q10.item()}, 90% quantile: {q90.item()}")
        
        # Use data between q10 and q90 for statistics
        filtered_data = band_data[(band_data >= q10) & (band_data <= q90)]
        
        print(f"  Filtered data shape: {filtered_data.shape}")
        print(f"  Filtered Min: {filtered_data.min().item()}, Max: {filtered_data.max().item()}")
        print(f"  Filtered Mean: {filtered_data.mean().item()}, Std: {filtered_data.std().item()}")
        
        sentinel_means[band] = torch.mean(filtered_data)
        sentinel_stds[band] = torch.std(filtered_data)
    
    # Calculate statistics for angles (already applied cosine)
    angles_means = torch.mean(torch.cat([angles_x_tensor, angles_y_tensor], dim=0), dim=0)
    angles_stds = torch.std(torch.cat([angles_x_tensor, angles_y_tensor], dim=0), dim=0)
    
    # Save statistics as .pth files
    torch.save(landsat_means, os.path.join(output_dir, 'mean_landsat8.pth'))
    torch.save(landsat_stds, os.path.join(output_dir, 'std_landsat8.pth'))
    torch.save(sentinel_means, os.path.join(output_dir, 'mean_sentinel2.pth'))
    torch.save(sentinel_stds, os.path.join(output_dir, 'std_sentinel2.pth'))
    torch.save(angles_means, os.path.join(output_dir, 'angles_mean.pth'))
    torch.save(angles_stds, os.path.join(output_dir, 'angles_std.pth'))
    
    # Print statistics summary
    print("\nStatistics Summary:")
    print(f"Landsat mean: {landsat_means}")
    print(f"Landsat std: {landsat_stds}")
    print(f"Sentinel mean: {sentinel_means}")
    print(f"Sentinel std: {sentinel_stds}")
    print(f"Angles mean (cosine): {angles_means}")
    print(f"Angles std (cosine): {angles_stds}")
    
    print(f"\nStatistics saved to {output_dir}")

if __name__ == "__main__":
    tensor_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data_pth'
    calculate_statistics(tensor_dir)