import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralResponseLayer(nn.Module):
    """
    Layer that applies Relative Spectral Response (RSR) functions to hyperspectral data.
    
    This layer transforms a hyperspectral representation into the specific bands
    of a satellite sensor by applying the sensor's RSR functions.
    
    Args:
        rsr_matrix (torch.Tensor or numpy.ndarray): Matrix of RSR values with shape 
            [output_channels, hyperspectral_dim]
        trainable (bool): Whether the RSR matrix should be trainable (learnable)
    """
    def __init__(self, rsr_matrix, trainable=False):
        super(SpectralResponseLayer, self).__init__()
        
        # Convert numpy array to tensor if needed
        if isinstance(rsr_matrix, np.ndarray):
            rsr_matrix = torch.from_numpy(rsr_matrix).float()
        
        # Register the RSR matrix as a parameter or buffer
        if trainable:
            self.rsr_matrix = nn.Parameter(rsr_matrix)
        else:
            self.register_buffer('rsr_matrix', rsr_matrix)
            
        # Store shapes for reference
        self.output_channels, self.hyperspectral_dim = rsr_matrix.shape
    
    @classmethod
    def from_rsr_file(cls, rsr_reader, rsr_type='vnir', trainable=False):
        """
        Create a SpectralResponseLayer from an RSRReader instance.
        
        Args:
            rsr_reader: An instance of the RSRReader class
            rsr_type (str): Type of RSR to use, either 'vnir', 'swir', or 'full'
            trainable (bool): Whether the RSR matrix should be trainable
            
        Returns:
            SpectralResponseLayer: Initialized with the appropriate RSR matrix
        """
        # Read RSR data from the reader
        full_matrix, vnir_matrix, swir_matrix, _, _, _ = rsr_reader.read_rsr_file()
        
        # Select the appropriate matrix based on type
        if rsr_type.lower() == 'vnir':
            rsr_matrix = torch.from_numpy(vnir_matrix).float()
        elif rsr_type.lower() == 'swir':
            rsr_matrix = torch.from_numpy(swir_matrix).float()
        elif rsr_type.lower() == 'full':
            rsr_matrix = torch.from_numpy(full_matrix).float()
        else:
            raise ValueError(f"Unknown RSR type: {rsr_type}. Use 'vnir', 'swir', or 'full'.")
        
        return cls(rsr_matrix, trainable)
    
    def forward(self, x):
        """
        Apply RSR functions to hyperspectral representation.
        
        Args:
            x (torch.Tensor): Tensor of shape [B, hyperspectral_dim, H, W]
            
        Returns:
            torch.Tensor: Tensor of shape [B, output_channels, H, W]
        """
        B, C, H, W = x.size()
        
        # Print shapes for debugging
        print(f"RSR input shape: {x.shape}")
        print(f"RSR matrix shape: {self.rsr_matrix.shape}")
        print(f"Expected hyperspectral_dim: {self.hyperspectral_dim}")
        
        # Ensure the input has the right number of channels
        if C != self.hyperspectral_dim:
            raise ValueError(f"Input has {C} channels but RSR matrix expects {self.hyperspectral_dim}")
        
        # Reshape for matrix multiplication:
        # [B, hyperspectral_dim, H, W] -> [B, hyperspectral_dim, H*W]
        x_reshaped = x.view(B, C, -1)
        
        # Correct way to apply the RSR matrix to convert from hyperspectral to satellite bands
        # self.rsr_matrix has shape [output_channels, hyperspectral_dim]
        # x_reshaped has shape [B, hyperspectral_dim, H*W]
        
        # Reshape x for matrix multiplication with the RSR matrix
        # [B, hyperspectral_dim, H*W] -> [B, H*W, hyperspectral_dim]
        x_transposed = x_reshaped.transpose(1, 2)
        
        # Apply RSR matrix: [B, H*W, hyperspectral_dim] × [output_channels, hyperspectral_dim]^T
        # Result: [B, H*W, output_channels]
        out = torch.matmul(x_transposed, self.rsr_matrix.t())
        
        # Transpose back: [B, H*W, output_channels] -> [B, output_channels, H*W]
        out = out.transpose(1, 2)
        
        # Reshape back: [B, output_channels, H*W] -> [B, output_channels, H, W]
        out = out.view(B, self.output_channels, H, W)
        
        return out


# Example usage:
if __name__ == "__main__":
    # Simulate an RSR matrix for 4 output bands from 100 hyperspectral bands
    rsr_matrix = torch.rand(4, 100)  # [output_channels, hyperspectral_dim]
    
    # Normalize each row to sum to 1 (typical for RSR functions)
    rsr_matrix = F.normalize(rsr_matrix, p=1, dim=1)
    
    # Create the SpectralResponseLayer
    rsr_layer = SpectralResponseLayer(rsr_matrix, trainable=False)
    
    # Create dummy hyperspectral input
    hyperspectral_data = torch.randn(2, 100, 32, 32)  # [batch_size, hyperspectral_dim, height, width]
    
    # Apply RSR functions
    output = rsr_layer(hyperspectral_data)
    
    # Print shapes
    print(f"Input hyperspectral data shape: {hyperspectral_data.shape}")
    print(f"Output satellite bands shape: {output.shape}")
    print(f"Expected output shape: [2, 4, 32, 32]")