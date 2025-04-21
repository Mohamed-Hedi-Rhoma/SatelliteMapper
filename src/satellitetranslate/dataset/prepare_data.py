import os
import glob
import json
import torch
import numpy as np
import rasterio
from tqdm import tqdm
import random
import shutil
import gc

# Define directories
DATA_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data_prepared"
OUTPUT_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/pth_data"

# Define bands
LANDSAT_BANDS = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir1.tif', 'swir2.tif']
SENTINEL_10M_BANDS = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif']
SENTINEL_20M_BANDS = ['swir1.tif', 'swir2.tif']

# Define target sizes
LANDSAT_SIZE = (128, 128)
SENTINEL_10M_SIZE = (384, 384)
SENTINEL_20M_SIZE = (192, 192)

# Define split ratios
TRAIN_RATIO = 0.85
VALID_RATIO = 0.10
TEST_RATIO = 0.05  # remaining 5%

# Define batch size for memory-friendly processing
BATCH_SIZE = 25  # Further reduced to avoid memory issues

def read_angles_json(json_path):
    """Read angles from JSON file and return as a list of 4 values: SZA, VZA, SAA, VAA."""
    try:
        with open(json_path, 'r') as f:
            angles_data = json.load(f)
        
        # Create a list of the 4 angle values, using 0.0 as a fallback if any are missing
        angles = [
            angles_data.get("SZA", 0.0) if angles_data.get("SZA") is not None else 0.0,
            angles_data.get("VZA", 0.0) if angles_data.get("VZA") is not None else 0.0,
            angles_data.get("SAA", 0.0) if angles_data.get("SAA") is not None else 0.0,
            angles_data.get("VAA", 0.0) if angles_data.get("VAA") is not None else 0.0
        ]
        return angles
    except Exception as e:
        print(f"Error reading angles from {json_path}: {e}")
        # Return zeros if there's an error
        return [0.0, 0.0, 0.0, 0.0]

def read_and_process_image(file_path, target_size):
    """
    Read an image and process it to the target size.
    Only crop if the image exceeds the target size.
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            
            h, w = data.shape
            target_h, target_w = target_size
            
            # Only crop if the image exceeds the target size
            if h > target_h or w > target_w:
                # Calculate crop coordinates to center the crop
                start_h = max(0, (h - target_h) // 2)
                start_w = max(0, (w - target_w) // 2)
                end_h = min(h, start_h + target_h)
                end_w = min(w, start_w + target_w)
                
                # Crop the image
                data = data[start_h:end_h, start_w:end_w]
                
                # If the cropped image is still smaller than target (shouldn't happen if properly cropped)
                if data.shape[0] < target_h or data.shape[1] < target_w:
                    # Create a zero-filled array of target size
                    padded_data = np.zeros(target_size, dtype=np.float32)
                    # Place the data in the center
                    pad_h = (target_h - data.shape[0]) // 2
                    pad_w = (target_w - data.shape[1]) // 2
                    padded_data[pad_h:pad_h+data.shape[0], pad_w:pad_w+data.shape[1]] = data
                    data = padded_data
            elif h < target_h or w < target_w:
                # If the image is smaller than target size, pad with zeros
                padded_data = np.zeros(target_size, dtype=np.float32)
                # Place the data in the center
                pad_h = (target_h - h) // 2
                pad_w = (target_w - w) // 2
                padded_data[pad_h:pad_h+h, pad_w:pad_w+w] = data
                data = padded_data
                
            return data
    except Exception as e:
        print(f"Error reading image {file_path}: {e}")
        # Return zeros if there's an error
        return np.zeros(target_size, dtype=np.float32)

def collect_data_paths():
    """
    Collect paths to all data points, ensuring consistent ordering across datasets.
    Returns a list of tuples: (landsat_dir, sentinel_dir, site_name, acq_date)
    """
    data_paths = []
    
    # Get all site directories
    site_dirs = [d for d in glob.glob(os.path.join(DATA_DIR, "*")) 
                if os.path.isdir(d) and not d.endswith("__pycache__")]
    
    for site_dir in sorted(site_dirs):
        site_name = os.path.basename(site_dir)
        
        # Get all acquisition date directories
        acq_dirs = [d for d in glob.glob(os.path.join(site_dir, "*")) 
                    if os.path.isdir(d) and not d.endswith("__pycache__")]
        
        for acq_dir in sorted(acq_dirs):
            acq_date = os.path.basename(acq_dir)
            
            # Define paths to Landsat and Sentinel data
            landsat_dir = os.path.join(acq_dir, "landsat8")
            sentinel_dir = os.path.join(acq_dir, "sentinel2")
            
            # Check if both directories exist
            if os.path.exists(landsat_dir) and os.path.exists(sentinel_dir):
                # Verify that all required band files exist for Landsat
                landsat_bands_exist = all(os.path.exists(os.path.join(landsat_dir, band)) for band in LANDSAT_BANDS)
                
                # Verify that all required band files exist for Sentinel
                sentinel_10m_bands_exist = all(os.path.exists(os.path.join(sentinel_dir, band)) for band in SENTINEL_10M_BANDS)
                sentinel_20m_bands_exist = all(os.path.exists(os.path.join(sentinel_dir, band)) for band in SENTINEL_20M_BANDS)
                
                # Only add if all required files exist
                if landsat_bands_exist and sentinel_10m_bands_exist and sentinel_20m_bands_exist:
                    data_paths.append((landsat_dir, sentinel_dir, site_name, acq_date))
                else:
                    missing = []
                    if not landsat_bands_exist:
                        missing.append("Landsat bands")
                    if not sentinel_10m_bands_exist:
                        missing.append("Sentinel 10m bands")
                    if not sentinel_20m_bands_exist:
                        missing.append("Sentinel 20m bands")
                    print(f"Skipping {site_name}/{acq_date} - Missing: {', '.join(missing)}")
    
    return data_paths

def process_mini_batch(data_paths_batch, split_name, batch_idx):
    """
    Process a mini-batch of data paths and save to separate .pth files.
    """
    # Initialize data arrays for this mini-batch
    landsat_data = []
    sentinel_10m_data = []
    sentinel_20m_data = []
    landsat_angles = []
    sentinel_angles = []
    
    # Process each data point in this mini-batch
    for landsat_dir, sentinel_dir, site_name, acq_date in tqdm(data_paths_batch, 
                                                            desc=f"Processing {split_name} batch {batch_idx}"):
        try:
            # Read Landsat bands
            landsat_bands = []
            for band in LANDSAT_BANDS:
                band_path = os.path.join(landsat_dir, band)
                band_data = read_and_process_image(band_path, LANDSAT_SIZE)
                landsat_bands.append(band_data)
            
            # Read Sentinel 10m bands
            sentinel_10m_bands = []
            for band in SENTINEL_10M_BANDS:
                band_path = os.path.join(sentinel_dir, band)
                band_data = read_and_process_image(band_path, SENTINEL_10M_SIZE)
                sentinel_10m_bands.append(band_data)
            
            # Read Sentinel 20m bands
            sentinel_20m_bands = []
            for band in SENTINEL_20M_BANDS:
                band_path = os.path.join(sentinel_dir, band)
                band_data = read_and_process_image(band_path, SENTINEL_20M_SIZE)
                sentinel_20m_bands.append(band_data)
            
            # Read angles
            landsat_angles_path = os.path.join(landsat_dir, "angles.json")
            sentinel_angles_path = os.path.join(sentinel_dir, "angles.json")
            
            landsat_angles_data = read_angles_json(landsat_angles_path)
            sentinel_angles_data = read_angles_json(sentinel_angles_path)
            
            # Append to lists
            landsat_data.append(np.stack(landsat_bands))
            sentinel_10m_data.append(np.stack(sentinel_10m_bands))
            sentinel_20m_data.append(np.stack(sentinel_20m_bands))
            landsat_angles.append(landsat_angles_data)
            sentinel_angles.append(sentinel_angles_data)
        except Exception as e:
            print(f"Error processing {site_name}/{acq_date}: {e}")
            continue
    
    # Convert lists to tensors
    print(f"Converting {split_name} batch {batch_idx} to tensors...")
    
    # Handle empty batches
    if not landsat_data:
        print(f"Warning: Empty batch for {split_name} batch {batch_idx}")
        return
    
    landsat_tensor = torch.tensor(np.stack(landsat_data))
    sentinel_10m_tensor = torch.tensor(np.stack(sentinel_10m_data))
    sentinel_20m_tensor = torch.tensor(np.stack(sentinel_20m_data))
    landsat_angles_tensor = torch.tensor(np.stack(landsat_angles))
    sentinel_angles_tensor = torch.tensor(np.stack(sentinel_angles))
    
    print(f"{split_name} batch {batch_idx} shapes:")
    print(f"Landsat data: {landsat_tensor.shape}")
    print(f"Sentinel 10m data: {sentinel_10m_tensor.shape}")
    print(f"Sentinel 20m data: {sentinel_20m_tensor.shape}")
    print(f"Landsat angles: {landsat_angles_tensor.shape}")
    print(f"Sentinel angles: {sentinel_angles_tensor.shape}")
    
    # Save tensors to .pth files with batch index
    batch_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(batch_dir, exist_ok=True)
    
    torch.save(landsat_tensor, os.path.join(batch_dir, f"data_x_{batch_idx}.pth"))
    torch.save(sentinel_10m_tensor, os.path.join(batch_dir, f"data_y1_{batch_idx}.pth"))
    torch.save(sentinel_20m_tensor, os.path.join(batch_dir, f"data_y2_{batch_idx}.pth"))
    torch.save(landsat_angles_tensor, os.path.join(batch_dir, f"angles_x_{batch_idx}.pth"))
    torch.save(sentinel_angles_tensor, os.path.join(batch_dir, f"angles_y_{batch_idx}.pth"))
    
    # Clear memory
    del landsat_data, sentinel_10m_data, sentinel_20m_data, landsat_angles, sentinel_angles
    del landsat_tensor, sentinel_10m_tensor, sentinel_20m_tensor, landsat_angles_tensor, sentinel_angles_tensor
    gc.collect()
    
    print(f"{split_name} batch {batch_idx} .pth files created successfully")

def process_data_split(data_paths, split_name):
    """Process all data for a split in mini-batches to avoid memory issues."""
    num_samples = len(data_paths)
    
    # Calculate number of batches
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE  # Ceiling division
    
    print(f"Processing {split_name} data: {num_samples} samples in {num_batches} batches")
    
    # Process mini-batches
    for i in range(num_batches):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, num_samples)
        
        batch_paths = data_paths[start_idx:end_idx]
        process_mini_batch(batch_paths, split_name, i)
        
        # Force garbage collection after each batch
        gc.collect()
    
    # Save index file to record the total number of batches
    index_info = {
        "num_batches": num_batches,
        "total_samples": num_samples,
        "batch_size": BATCH_SIZE
    }
    
    with open(os.path.join(OUTPUT_DIR, split_name, "index.json"), 'w') as f:
        json.dump(index_info, f, indent=4)
    
    print(f"Completed processing {split_name} data")

def merge_data_type_files(split_name, data_type, shape_info=None):
    """
    Merge all batch files for a specific data type (e.g. 'data_x', 'data_y1').
    Process one data type at a time to reduce memory usage.
    """
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    
    if not os.path.exists(split_dir):
        print(f"Directory {split_dir} does not exist. Skipping {data_type} merge.")
        return False
    
    # Get all batch files for this data type
    batch_files = sorted(glob.glob(os.path.join(split_dir, f"{data_type}_*.pth")))
    
    if not batch_files:
        print(f"No batch files found for {data_type} in {split_name}. Skipping.")
        return False
    
    print(f"Merging {len(batch_files)} batch files for {data_type} in {split_name}...")
    
    # Initialize an empty tensor list
    all_tensors = []
    total_samples = 0
    
    # Load and append each batch
    for i, batch_file in enumerate(batch_files):
        try:
            print(f"Loading {data_type} batch {i}/{len(batch_files)-1} for {split_name}...")
            tensor = torch.load(batch_file)
            total_samples += tensor.shape[0]
            all_tensors.append(tensor)
            
            # If shape info is provided, verify the tensor shape (excluding batch dimension)
            if shape_info and len(tensor.shape) > 1:
                expected_shape = shape_info[1:]  # Skip the batch dimension
                actual_shape = tensor.shape[1:]
                if actual_shape != tuple(expected_shape):
                    print(f"Warning: {os.path.basename(batch_file)} has unexpected shape {tensor.shape}, expected [..., {', '.join(map(str, expected_shape))}]")
            
            # Force garbage collection after loading each batch
            gc.collect()
        except Exception as e:
            print(f"Error loading {batch_file}: {e}")
            # Continue with other batches even if one fails
            continue
    
    if not all_tensors:
        print(f"No tensors loaded for {data_type} in {split_name}. Skipping merge.")
        return False
    
    try:
        # Concatenate and save
        print(f"Concatenating {len(all_tensors)} tensors for {data_type} in {split_name}...")
        merged_tensor = torch.cat(all_tensors, dim=0)
        print(f"  Merged shape: {merged_tensor.shape}")
        
        # Save merged file
        output_file = os.path.join(OUTPUT_DIR, f"{data_type}_{split_name}.pth")
        print(f"  Saving to {output_file}...")
        torch.save(merged_tensor, output_file)
        
        # Verify file was created successfully
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            print(f"  Successfully saved {output_file}")
            
            # Clear memory
            del merged_tensor
            del all_tensors
            gc.collect()
            
            return True
        else:
            print(f"  Failed to save {output_file} properly")
            return False
    except Exception as e:
        print(f"Error merging {data_type} in {split_name}: {e}")
        
        # If we get a specific error related to file size or memory, try saving in smaller chunks
        if "file size" in str(e).lower() or "memory" in str(e).lower():
            try:
                print(f"Attempting alternative approach with torch.save...")
                # Try using a different approach to save
                output_file = os.path.join(OUTPUT_DIR, f"{data_type}_{split_name}.pth")
                merged_tensor = torch.cat(all_tensors, dim=0)
                
                # Use lower-level save operation that might reduce memory usage
                with open(output_file, 'wb') as f:
                    torch.save(merged_tensor, f)
                
                # Verify file was created
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"  Successfully saved {output_file} with alternative method")
                    
                    # Clear memory
                    del merged_tensor
                    del all_tensors
                    gc.collect()
                    
                    return True
            except Exception as inner_e:
                print(f"  Alternative saving method also failed: {inner_e}")
        
        # Clear memory before returning
        gc.collect()
        return False

def merge_batches(split_name):
    """
    Merge all batch files for a split into a single file.
    Process one data type at a time to reduce memory usage.
    """
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    
    if not os.path.exists(split_dir):
        print(f"Directory {split_dir} does not exist. Skipping {split_name} merge.")
        return False
    
    # Load index file to determine expected shapes
    expected_shapes = {}
    try:
        with open(os.path.join(split_dir, "index.json"), 'r') as f:
            index_info = json.load(f)
            num_samples = index_info["total_samples"]
            
            # Set expected shapes based on known dimensions
            expected_shapes["data_x"] = [num_samples, 6, 128, 128]
            expected_shapes["data_y1"] = [num_samples, 4, 384, 384]
            expected_shapes["data_y2"] = [num_samples, 2, 192, 192]
            expected_shapes["angles_x"] = [num_samples, 4]
            expected_shapes["angles_y"] = [num_samples, 4]
    except Exception:
        # If index file doesn't exist or has issues, proceed without expected shapes
        expected_shapes = None
    
    # Process each data type separately to reduce memory usage
    data_types = ["data_x", "data_y1", "data_y2", "angles_x", "angles_y"]
    success = True
    
    for data_type in data_types:
        shape_info = expected_shapes.get(data_type) if expected_shapes else None
        type_success = merge_data_type_files(split_name, data_type, shape_info)
        
        # If any data type fails, mark the overall merge as failed
        if not type_success:
            success = False
        
        # Force garbage collection after each data type
        gc.collect()
    
    return success

def cleanup_batch_files(split_name):
    """Delete all batch files after merging."""
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    
    if not os.path.exists(split_dir):
        print(f"Directory {split_name} does not exist. Skipping cleanup.")
        return
    
    print(f"Checking merged files for {split_name} before cleaning up...")
    
    # First check if the merged files exist and have valid sizes
    merged_files = {
        "data_x": os.path.join(OUTPUT_DIR, f"data_x_{split_name}.pth"),
        "data_y1": os.path.join(OUTPUT_DIR, f"data_y1_{split_name}.pth"),
        "data_y2": os.path.join(OUTPUT_DIR, f"data_y2_{split_name}.pth"),
        "angles_x": os.path.join(OUTPUT_DIR, f"angles_x_{split_name}.pth"),
        "angles_y": os.path.join(OUTPUT_DIR, f"angles_y_{split_name}.pth")
    }
    
    all_files_valid = True
    
    # Check each merged file
    for data_type, file_path in merged_files.items():
        if not os.path.exists(file_path):
            print(f"  Missing merged file: {os.path.basename(file_path)}")
            all_files_valid = False
            continue
        
        # Check file size (should be non-zero)
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if file_size < 0.01:  # Very small files might be empty/corrupted
            print(f"  Suspiciously small merged file: {os.path.basename(file_path)} ({file_size:.2f} MB)")
            all_files_valid = False
            continue
        
        # Try to load the file to verify it's valid
        try:
            # Just load the metadata without loading the full tensor into memory
            torch_file = torch.load(file_path, map_location=torch.device('cpu'))
            if not isinstance(torch_file, torch.Tensor):
                print(f"  File {os.path.basename(file_path)} doesn't contain a tensor")
                all_files_valid = False
            elif torch_file.numel() == 0:
                print(f"  File {os.path.basename(file_path)} contains an empty tensor")
                all_files_valid = False
            else:
                print(f"  Valid merged file: {os.path.basename(file_path)} ({file_size:.2f} MB)")
            
            # Clear memory
            del torch_file
            gc.collect()
        except Exception as e:
            print(f"  Error validating {os.path.basename(file_path)}: {e}")
            all_files_valid = False
    
    # Only delete if all merged files are valid
    if all_files_valid:
        print(f"All merged files for {split_name} are valid. Cleaning up batch files...")
        
        # Remove all batch files
        batch_files = glob.glob(os.path.join(split_dir, "*.pth"))
        for file in batch_files:
            try:
                os.remove(file)
                print(f"  Deleted {os.path.basename(file)}")
            except Exception as e:
                print(f"  Error deleting {file}: {e}")
        
        # Remove index.json
        try:
            index_file = os.path.join(split_dir, "index.json")
            if os.path.exists(index_file):
                os.remove(index_file)
                print(f"  Deleted {os.path.basename(index_file)}")
        except Exception as e:
            print(f"  Error deleting index file: {e}")
        
        # Remove the directory
        try:
            os.rmdir(split_dir)
            print(f"  Removed directory {split_dir}")
        except Exception as e:
            print(f"  Error removing directory {split_dir}: {e}")
    else:
        print(f"Not all merged files for {split_name} are valid. Keeping batch files for safety.")

def create_pth_files():
    """
    Create the .pth files for training, validation, and testing.
    Using mini-batches to avoid memory issues. Then merge all batches
    and clean up batch files.
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Collect all paths to data
    print("Collecting data paths...")
    data_paths = collect_data_paths()
    total_samples = len(data_paths)
    print(f"Found {total_samples} complete data points")
    
    # Shuffle data paths to ensure random distribution
    random.shuffle(data_paths)
    
    # Calculate split indices
    train_size = int(total_samples * TRAIN_RATIO)
    valid_size = int(total_samples * VALID_RATIO)
    
    # Create splits
    train_paths = data_paths[:train_size]
    valid_paths = data_paths[train_size:train_size + valid_size]
    test_paths = data_paths[train_size + valid_size:]
    
    print(f"Split sizes: Train={len(train_paths)}, Valid={len(valid_paths)}, Test={len(test_paths)}")
    
    # Create directories for each split
    os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "valid"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "test"), exist_ok=True)
    
    # Create a metadata file with split info
    split_info = {
        "train_samples": len(train_paths),
        "valid_samples": len(valid_paths),
        "test_samples": len(test_paths),
        "total_samples": total_samples,
        "train_ratio": TRAIN_RATIO,
        "valid_ratio": VALID_RATIO,
        "test_ratio": TEST_RATIO
    }
    
    with open(os.path.join(OUTPUT_DIR, "splits.json"), 'w') as f:
        json.dump(split_info, f, indent=4)
    
    # Process and save each split separately in mini-batches
    print("\n=== PROCESSING TRAINING DATA ===")
    process_data_split(train_paths, "train")
    
    print("\n=== PROCESSING VALIDATION DATA ===")
    process_data_split(valid_paths, "valid")
    
    print("\n=== PROCESSING TEST DATA ===")
    process_data_split(test_paths, "test")
    
    print("\n=== MERGING BATCHES ===")
    # Merge batches and clean up
    for split in ["train", "valid", "test"]:
        print(f"\nProcessing {split} split...")
        merge_success = merge_batches(split)
        
        # Only cleanup if merge was successful
        if merge_success:
            print(f"Merge for {split} was successful. Cleaning up batch files.")
            cleanup_batch_files(split)
        else:
            print(f"Merge for {split} had issues. Keeping batch files.")
    
    print("\nAll data processing completed.")
    print(f"Files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    create_pth_files()