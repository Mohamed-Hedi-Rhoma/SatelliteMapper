import os
import glob
import json
import torch
import shutil

# Define output directory
OUTPUT_DIR = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/pth_data"

def merge_batches_in_directory(split_name):
    """
    Merge all batch files for a split into a single file.
    Modified to handle partial merges and require less memory.
    """
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    
    if not os.path.exists(split_dir):
        print(f"Directory {split_dir} does not exist. Skipping.")
        return False

    # Check if any batch files exist in the directory
    batch_files = glob.glob(os.path.join(split_dir, "data_x_*.pth"))
    if not batch_files:
        print(f"No batch files found in {split_dir}. Skipping.")
        return False

    num_batches = len(batch_files)
    print(f"Found {num_batches} batches in {split_dir}")
    
    # Define output file paths
    output_files = {
        "data_x": os.path.join(OUTPUT_DIR, f"data_x_{split_name}.pth"),
        "data_y1": os.path.join(OUTPUT_DIR, f"data_y1_{split_name}.pth"),
        "data_y2": os.path.join(OUTPUT_DIR, f"data_y2_{split_name}.pth"),
        "angles_x": os.path.join(OUTPUT_DIR, f"angles_x_{split_name}.pth"),
        "angles_y": os.path.join(OUTPUT_DIR, f"angles_y_{split_name}.pth")
    }
    
    # Process each type of data separately to reduce memory usage
    for data_type in ["data_x", "data_y1", "data_y2", "angles_x", "angles_y"]:
        print(f"Processing {data_type} for {split_name}...")
        
        all_tensors = []
        
        # Load all batch files for this data type
        for i in range(num_batches):
            batch_file = os.path.join(split_dir, f"{data_type}_{i}.pth")
            
            if os.path.exists(batch_file):
                try:
                    tensor = torch.load(batch_file)
                    all_tensors.append(tensor)
                    print(f"  Loaded {data_type} batch {i} with shape {tensor.shape}")
                except Exception as e:
                    print(f"  Error loading {batch_file}: {e}")
        
        if all_tensors:
            try:
                # Concatenate tensors and save
                merged_tensor = torch.cat(all_tensors, dim=0)
                print(f"  Merged {data_type} shape: {merged_tensor.shape}")
                
                torch.save(merged_tensor, output_files[data_type])
                print(f"  Saved merged {data_type} to {output_files[data_type]}")
                
                # Free memory
                del merged_tensor
                del all_tensors
            except Exception as e:
                print(f"  Error merging {data_type}: {e}")
                return False
        else:
            print(f"  No tensors found for {data_type}")
    
    print(f"Successfully merged all tensors for {split_name}")
    return True

def move_files_from_subdir(split_name):
    """
    Check if there are already merged files in the split directory
    and move them to the main output directory if needed.
    """
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    
    if not os.path.exists(split_dir):
        return
    
    # Look for merged files in the split directory
    file_patterns = [
        f"data_x_{split_name}.pth",
        f"data_y1_{split_name}.pth", 
        f"data_y2_{split_name}.pth",
        f"angles_x_{split_name}.pth",
        f"angles_y_{split_name}.pth"
    ]
    
    for pattern in file_patterns:
        file_path = os.path.join(split_dir, pattern)
        if os.path.exists(file_path):
            target_path = os.path.join(OUTPUT_DIR, pattern)
            try:
                print(f"Moving {file_path} to {target_path}")
                shutil.move(file_path, target_path)
            except Exception as e:
                print(f"Error moving {file_path}: {e}")

def cleanup_directory(directory):
    """Clean up a directory and all its contents."""
    if not os.path.exists(directory):
        return
    
    try:
        shutil.rmtree(directory)
        print(f"Removed directory {directory}")
    except Exception as e:
        print(f"Error removing directory {directory}: {e}")
        
        # Try deleting files one by one if rmtree fails
        try:
            for file in glob.glob(os.path.join(directory, "*")):
                try:
                    if os.path.isfile(file):
                        os.remove(file)
                    elif os.path.isdir(file):
                        shutil.rmtree(file)
                except Exception as e:
                    print(f"Error removing {file}: {e}")
        except Exception:
            pass

def main():
    """Main function to merge batches and clean up."""
    print("Starting to merge and clean up data...")
    
    for split_name in ["train", "valid", "test"]:
        print(f"\nProcessing {split_name} split:")
        
        # First check if any files need to be moved from subdirectories
        move_files_from_subdir(split_name)
        
        # Then try to merge any remaining batch files
        if merge_batches_in_directory(split_name):
            print(f"Merged {split_name} data successfully")
        else:
            print(f"Some issues occurred during {split_name} merge")
            
        # Clean up the directory regardless of merge success
        cleanup_directory(os.path.join(OUTPUT_DIR, split_name))
    
    print("\nAll operations completed!")
    
    # Print summary of files in output directory
    print("\nFinal files in output directory:")
    for file in sorted(os.listdir(OUTPUT_DIR)):
        file_path = os.path.join(OUTPUT_DIR, file)
        if os.path.isfile(file_path):
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"  {file} ({file_size:.2f} MB)")

if __name__ == "__main__":
    main()