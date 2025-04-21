import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """
    Self-attention block for feature maps.
    
    This block implements a spatial self-attention mechanism that allows 
    the model to focus on important regions in the feature maps.
    
    Args:
        channels (int): Number of input channels
        reduction_ratio (int): Channel reduction ratio for the query and key projections
    """
    def __init__(self, channels, reduction_ratio=8):
        super(AttentionBlock, self).__init__()
        
        # Reduced dimension for query and key projections (for efficiency)
        self.reduced_dim = channels // reduction_ratio
        
        # Query, key, and value projections
        self.query = nn.Conv2d(channels, self.reduced_dim, kernel_size=1)
        self.key = nn.Conv2d(channels, self.reduced_dim, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Learnable scaling parameter for the attention map
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass of the attention block.
        
        Args:
            x (torch.Tensor): Input feature map of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Attention-enhanced feature map of shape [B, C, H, W]
        """
        batch_size, C, H, W = x.size()
        
        # Project input to query
        # Shape: [B, C//r, H, W] where r is the reduction ratio
        proj_query = self.query(x)
        
        # Reshape query for matrix multiplication
        # Shape: [B, H*W, C//r]
        proj_query = proj_query.view(batch_size, self.reduced_dim, -1).permute(0, 2, 1)
        
        # Project input to key
        # Shape: [B, C//r, H, W]
        proj_key = self.key(x)
        
        # Reshape key for matrix multiplication
        # Shape: [B, C//r, H*W]
        proj_key = proj_key.view(batch_size, self.reduced_dim, -1)
        
        # Calculate attention map
        # Shape: [B, H*W, H*W]
        energy = torch.bmm(proj_query, proj_key)
        
        # Apply softmax to normalize attention weights
        # Each row sums to 1, representing attention weights for one position
        attention = F.softmax(energy, dim=2)
        
        # Project input to value
        # Shape: [B, C, H, W]
        proj_value = self.value(x)
        
        # Reshape value for matrix multiplication
        # Shape: [B, C, H*W]
        proj_value = proj_value.view(batch_size, C, -1)
        
        # Apply attention weights to values
        # Shape: [B, C, H*W]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        # Reshape back to feature map
        # Shape: [B, C, H, W]
        out = out.view(batch_size, C, H, W)
        
        # Residual connection with learnable weight
        # At initialization, gamma is 0, so the attention has no effect
        # During training, the model learns how much attention to apply
        out = self.gamma * out + x
        
        return out


# Example usage:
if __name__ == "__main__":
    # Create an attention block
    block = AttentionBlock(channels=64, reduction_ratio=8)
    
    # Create dummy input tensor
    features = torch.randn(2, 64, 32, 32)  # [batch_size, channels, height, width]
    
    # Forward pass
    output = block(features)
    
    # Print shapes
    print(f"Input shape: {features.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output should match input shape: {output.shape == features.shape}")