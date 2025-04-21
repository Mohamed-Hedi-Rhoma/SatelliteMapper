import os
import glob
import torch
import gc
import sys

def sequential_merge(input_dir, output_file, pattern, batch_size=5):
    """
    Sequentially merge batch files into a single output file.
    
    Args:
        input_dir: Directory containing batch files
        output_file: Path to the output merged file
        pattern: Glob pattern to match batch files (e.g., "data_y1_*.pth")
        batch_size: Number of batch files to merge at once
    """
    # Get all batch files matching the pattern
    batch_files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    if not batch_files:
        print(f"No files found matching pattern '{pattern}' in {input_dir}")
        return False
    
    print(f"Found {len(batch_files)} files to merge")
    
    # Initialize with the first batch file
    print(f"Initializing with first batch...")
    merged_tensor = None
    
    # Process batch files in smaller groups
    for i in range(0, len(batch_files), batch_size):
        group = batch_files[i:i+batch_size]
        print(f"Processing group {i//batch_size + 1}/{(len(batch_files) + batch_size - 1)//batch_size}: files {i+1}-{i+len(group)}")
        
        # Load and process this group
        group_tensors = []
        for batch_file in group:
            try:
                print(f"  Loading {os.path.basename(batch_file)}...")
                tensor = torch.load(batch_file)
                group_tensors.append(tensor)
                print(f"    Shape: {tensor.shape}")
            except Exception as e:
                print(f"  Error loading {batch_file}: {e}")
                continue
        
        # Skip if no valid tensors in this group
        if not group_tensors:
            print(f"  No valid tensors in group {i//batch_size + 1}, skipping")
            continue
        
        # Concatenate this group
        print(f"  Concatenating group...")
        group_tensor = torch.cat(group_tensors, dim=0)
        print(f"  Group tensor shape: {group_tensor.shape}")
        
        # If this is the first group, initialize merged_tensor
        if merged_tensor is None:
            merged_tensor = group_tensor
        else:
            # Otherwise, concatenate with existing merged_tensor
            print(f"  Adding to merged tensor (current shape: {merged_tensor.shape})...")
            merged_tensor = torch.cat([merged_tensor, group_tensor], dim=0)
        
        print(f"  Updated merged tensor shape: {merged_tensor.shape}")
        
        # Clear memory for this group
        del group_tensors, group_tensor
        gc.collect()
        
        # Save intermediate result to avoid losing progress
        intermediate_file = output_file + f".partial{i//batch_size + 1}"
        print(f"  Saving intermediate result to {os.path.basename(intermediate_file)}...")
        torch.save(merged_tensor, intermediate_file)
    
    # Save final result
    if merged_tensor is not None:
        print(f"Saving final merged tensor (shape: {merged_tensor.shape}) to {output_file}...")
        torch.save(merged_tensor, output_file)
        print(f"Successfully merged to {output_file}")
        
        # Clean up intermediate files
        for i in range(1, (len(batch_files) + batch_size - 1)//batch_size + 1):
            intermediate_file = output_file + f".partial{i}"
            if os.path.exists(intermediate_file):
                try:
                    os.remove(intermediate_file)
                    print(f"Removed intermediate file {os.path.basename(intermediate_file)}")
                except:
                    print(f"Failed to remove {os.path.basename(intermediate_file)}")
        
        return True
    else:
        print("Failed to create merged tensor")
        return False

def verify_merged_file(file_path):
    """Verify that the merged file exists and contains valid data."""
    if not os.path.exists(file_path):
        print(f"Merged file {file_path} does not exist")
        return False
    
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Merged file size: {file_size_mb:.2f} MB")
    
    try:
        # Try to load the tensor metadata without loading the full tensor
        tensor = torch.load(file_path, map_location="cpu")
        print(f"Tensor shape: {tensor.shape}")
        print(f"Tensor type: {tensor.dtype}")
        del tensor
        gc.collect()
        return True
    except Exception as e:
        print(f"Error verifying merged file: {e}")
        return False

def main():
    # Set your paths
    data_dir = "C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/pth_data"
    train_dir = os.path.join(data_dir, "train")
    output_file = os.path.join(data_dir, "data_y1_train.pth")
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        user_input = input(f"Output file {output_file} already exists. Overwrite? (y/n): ")
        if user_input.lower() != 'y':
            print("Aborting")
            return
    
    # Merge the files
    print(f"Starting merge process for data_y1_train.pth")
    success = sequential_merge(train_dir, output_file, "data_y1_*.pth", batch_size=3)
    
    if success:
        # Verify the merged file
        print("\nVerifying merged file...")
        verify_merged_file(output_file)
    
    print("\nDone!")

if __name__ == "__main__":
    # Print system info
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("")
    
    # Run the main function
    main()