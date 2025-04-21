import os
import torch
from torch.utils.data import Dataset, DataLoader

class SatelliteDataset(Dataset):
    """
    Dataset for satellite imagery with angle information.
    
    Loads Landsat 8 (x) and Sentinel-2 (y1, y2) data along with their angle information.
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
        self.landsat_data = torch.load(os.path.join(tensor_dir, f'data_x_{dataset_type}.pth'))
        self.sentinel_10m_data = torch.load(os.path.join(tensor_dir, f'data_y1_{dataset_type}.pth'))
        self.sentinel_20m_data = torch.load(os.path.join(tensor_dir, f'data_y2_{dataset_type}.pth'))
        
        # Load angle data
        self.landsat_angles = torch.load(os.path.join(tensor_dir, f'angles_x_{dataset_type}.pth'))
        self.sentinel_angles = torch.load(os.path.join(tensor_dir, f'angles_y_{dataset_type}.pth'))
        
        # Validate data shapes
        assert len(self.landsat_data) == len(self.sentinel_10m_data), "Landsat and Sentinel 10m data must have the same number of samples"
        assert len(self.landsat_data) == len(self.sentinel_20m_data), "Landsat and Sentinel 20m data must have the same number of samples"
        assert len(self.landsat_data) == len(self.landsat_angles), "Landsat data and angles must have the same number of samples"
        assert len(self.landsat_data) == len(self.sentinel_angles), "Landsat data and Sentinel angles must have the same number of samples"
        
        print(f"Loaded {dataset_type} dataset with {len(self.landsat_data)} samples")
        print(f"Landsat data shape: {self.landsat_data.shape}")
        print(f"Sentinel 10m data shape: {self.sentinel_10m_data.shape}")
        print(f"Sentinel 20m data shape: {self.sentinel_20m_data.shape}")
        print(f"Landsat angles shape: {self.landsat_angles.shape}")
        print(f"Sentinel angles shape: {self.sentinel_angles.shape}")
    
    def __len__(self):
        return len(self.landsat_data)
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Returns a dictionary containing:
        - 'landsat_img': Landsat 8 image tensor (6, 128, 128)
        - 'sentinel_10m_img': Sentinel-2 10m bands image tensor (4, 384, 384)
        - 'sentinel_20m_img': Sentinel-2 20m bands image tensor (2, 192, 192)
        - 'landsat_angles': Landsat 8 angle tensor (4,)
        - 'sentinel_angles': Sentinel-2 angle tensor (4,)
        """
        # Get images
        landsat_img = self.landsat_data[idx]
        sentinel_10m_img = self.sentinel_10m_data[idx]
        sentinel_20m_img = self.sentinel_20m_data[idx]
        
        # Get angles
        landsat_angles = self.landsat_angles[idx]
        sentinel_angles = self.sentinel_angles[idx]
        
        # Apply transformations if specified
        if self.transform:
            landsat_img = self.transform(landsat_img)
            sentinel_10m_img = self.transform(sentinel_10m_img)
            sentinel_20m_img = self.transform(sentinel_20m_img)
        
        return {
            'landsat_img': landsat_img,
            'sentinel_10m_img': sentinel_10m_img,
            'sentinel_20m_img': sentinel_20m_img,
            'landsat_angles': landsat_angles,
            'sentinel_angles': sentinel_angles
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
    tensor_dir = 'C:/Users/msi/Desktop/SatelliteTranslate/Satellitetranslate/pth_data'
    
    # Create dataloaders
    train_loader = create_train_dataloader(tensor_dir)
    valid_loader = create_valid_dataloader(tensor_dir)
    test_loader = create_test_dataloader(tensor_dir)
    
    # Print dataloader info
    print(f"\nTrain loader: {len(train_loader)} batches")
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