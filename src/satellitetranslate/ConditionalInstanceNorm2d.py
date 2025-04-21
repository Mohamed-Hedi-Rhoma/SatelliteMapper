import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionalInstanceNorm2d(nn.Module):
    """
    Conditional Instance Normalization layer.
    
    This layer performs instance normalization but with parameters (scale and bias)
    that are modulated by an external condition (in this case, angle information).
    
    Args:
        num_features (int): Number of input channels
        angle_dim (int): Dimension of the angle vector (e.g., 4 for solar/view azimuth/zenith)
        hidden_dim (int, optional): Hidden dimension for the angle encoder. Default: 128
    """
    def __init__(self, num_features, angle_dim, hidden_dim=128):
        super(ConditionalInstanceNorm2d, self).__init__()
        self.num_features = num_features
        
        # Apply instance normalization without learnable affine parameters
        # We'll generate these parameters from the angle information
        self.instance_norm = nn.InstanceNorm2d(num_features, affine=False)
        
        # Network to generate scale (gamma) and bias (beta) from angle information
        self.angle_encoder = nn.Sequential(
            nn.Linear(angle_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_features * 2)  # Scale and bias
        )
        
        # Initialize the angle encoder to produce normalized outputs
        self._initialize_parameters()
        
    def _initialize_parameters(self):
        """Initialize the angle encoder to produce unit scale and zero bias by default."""
        # Initialize the last layer to output zeros
        # This ensures that at the start of training, the conditional normalization
        # behaves similar to standard instance normalization
        nn.init.zeros_(self.angle_encoder[-1].weight)
        
        # Initialize the bias of the last layer to produce 1s for the scale (gamma)
        # and 0s for the bias (beta)
        nn.init.constant_(self.angle_encoder[-1].bias[:self.num_features], 1.0)  # gamma: initialize to 1
        nn.init.zeros_(self.angle_encoder[-1].bias[self.num_features:])  # beta: initialize to 0
        
    def forward(self, x, angles):
        """
        Forward pass of the conditional instance normalization.
        
        Args:
            x (torch.Tensor): Input feature map of shape [B, C, H, W]
            angles (torch.Tensor): Angle information of shape [B, angle_dim]
            
        Returns:
            torch.Tensor: Normalized feature map of shape [B, C, H, W]
        """
        # Apply instance normalization
        normalized = self.instance_norm(x)
        
        # Generate scale and bias from angle information
        params = self.angle_encoder(angles)
        
        # Split into scale (gamma) and bias (beta)
        gamma, beta = params.chunk(2, dim=1)
        
        # Reshape for broadcasting to [B, C, 1, 1]
        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        
        # Apply scale and bias
        output = gamma * normalized + beta
        
        return output