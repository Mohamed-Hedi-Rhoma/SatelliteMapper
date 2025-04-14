import os
import torch
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SatelliteDataset(Dataset):
    """
    Dataset for satellite imagery with angle information.
    
    Loads Landsat 8 (x) and Sentinel-2 (y) data along with their angle information.
    """
    def __init__(self, tensor_dir, dataset_type='train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            tensor_dir (str): Directory containing tensor and angle files
            dataset_type (str): One of 'train', 'valid', or 'test'
            transform (callable, optional): Optional transform to be applied to the images
        """
        self.tensor_dir = tensor_dir
        self.dataset_type = dataset_type
        self.transform = transform
        
        # Load tensor data
        self.x_data = torch.load(os.path.join(tensor_dir, f'{dataset_type}_x.pth'))
        self.y_data = torch.load(os.path.join(tensor_dir, f'{dataset_type}_y.pth'))
        
        # Load angle data
        with open(os.path.join(tensor_dir, f'angles_{dataset_type}_x.json'), 'r') as f:
            self.x_angles = json.load(f)
        
        with open(os.path.join(tensor_dir, f'angles_{dataset_type}_y.json'), 'r') as f:
            self.y_angles = json.load(f)
        
        # Validate data shapes
        assert len(self.x_data) == len(self.y_data), "X and Y data must have the same number of samples"
        assert len(self.x_data) == len(self.x_angles), "X data and X angles must have the same number of samples"
        assert len(self.y_data) == len(self.y_angles), "Y data and Y angles must have the same number of samples"
        
        # Convert angle dictionaries to tensors for easier batch processing
        self.x_angle_tensors = self._prepare_angle_tensors(self.x_angles)
        self.y_angle_tensors = self._prepare_angle_tensors(self.y_angles)
        
        print(f"Loaded {dataset_type} dataset with {len(self.x_data)} samples")
        print(f"X shape: {self.x_data.shape}, Y shape: {self.y_data.shape}")
    
    def _prepare_angle_tensors(self, angle_data):
        """
        Prepare angle data as tensors.
        
        Args:
            angle_data (list): List of dictionaries containing angle information
            
        Returns:
            torch.Tensor: Tensor of angle values with shape [n_samples, 4]
        """
        angle_tensors = torch.zeros((len(angle_data), 4), dtype=torch.float32)
        
        for i, sample in enumerate(angle_data):
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
            
            angle_tensors[i] = torch.tensor(angle_values, dtype=torch.float32)
        
        return angle_tensors
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Returns a dictionary containing:
        - 'landsat_img': Landsat 8 image tensor (6, 128, 128)
        - 'sentinel_img': Sentinel-2 image tensor (6, 384, 384)
        - 'landsat_angles': Landsat 8 angle tensor (4,)
        - 'sentinel_angles': Sentinel-2 angle tensor (4,)
        - 'landsat_site': Site name for Landsat image
        - 'sentinel_site': Site name for Sentinel image
        """
        # Get images
        landsat_img = self.x_data[idx]
        sentinel_img = self.y_data[idx]
        
        # Get angles
        landsat_angles = self.x_angle_tensors[idx]
        sentinel_angles = self.y_angle_tensors[idx]
        
        # Get site names for reference
        landsat_site = self.x_angles[idx]['site']
        sentinel_site = self.y_angles[idx]['site']
        
        # Apply transformations if specified
        if self.transform:
            landsat_img = self.transform(landsat_img)
            sentinel_img = self.transform(sentinel_img)
        
        return {
            'landsat_img': landsat_img,
            'sentinel_img': sentinel_img,
            'landsat_angles': landsat_angles,
            'sentinel_angles': sentinel_angles,
            'landsat_site': landsat_site,
            'sentinel_site': sentinel_site
        }

def create_train_dataloader(tensor_dir, batch_size=8, num_workers=4, shuffle=True):
    """
    Create a DataLoader for training data.
    
    Args:
        tensor_dir (str): Directory containing tensor and angle files
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for training data
    """
    # Define any training-specific transforms here
    train_transform = None  # You might add data augmentation here
    
    # Create dataset
    train_dataset = SatelliteDataset(
        tensor_dir=tensor_dir,
        dataset_type='train',
        transform=train_transform
    )
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader

def create_valid_dataloader(tensor_dir, batch_size=4, num_workers=4):
    """
    Create a DataLoader for validation data.
    
    Args:
        tensor_dir (str): Directory containing tensor and angle files
        batch_size (int): Batch size for validation
        num_workers (int): Number of workers for data loading
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for validation data
    """
    # Create dataset (no transforms for validation)
    valid_dataset = SatelliteDataset(
        tensor_dir=tensor_dir,
        dataset_type='valid',
        transform=None
    )
    
    # Create dataloader (no shuffling for validation)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return valid_loader

def create_test_dataloader(tensor_dir, batch_size=1, num_workers=4):
    """
    Create a DataLoader for test data.
    
    Args:
        tensor_dir (str): Directory containing tensor and angle files
        batch_size (int): Batch size for testing (usually 1 for full evaluation)
        num_workers (int): Number of workers for data loading
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for test data
    """
    # Create dataset (no transforms for testing)
    test_dataset = SatelliteDataset(
        tensor_dir=tensor_dir,
        dataset_type='test',
        transform=None
    )
    
    # Create dataloader (no shuffling for testing)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return test_loader

# Example usage
if __name__ == "__main__":
    tensor_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/data_pth'
    
    # Create dataloaders
    train_loader = create_train_dataloader(tensor_dir)
    valid_loader = create_valid_dataloader(tensor_dir)
    test_loader = create_test_dataloader(tensor_dir)
    
    # Print dataloader info
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Valid loader: {len(valid_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Get a batch to check the structure
    batch = next(iter(train_loader))
    print("\nBatch structure:")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {type(value)}")