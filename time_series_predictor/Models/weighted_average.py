import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedAverageModel(nn.Module):
    def __init__(self, timesteps, features):
        """
        Weighted Average Model where the weights across timesteps are learned.

        Parameters:
        - timesteps (int): Number of timesteps in the input sequence.
        - features (int): Number of features in each timestep.
        """
        super(WeightedAverageModel, self).__init__()
        
        # Trainable weights for the timesteps
        self.weights = nn.Parameter(torch.randn(timesteps))  # Initialized randomly
    
    def forward(self, x):
        """
        Forward pass of the weighted average model.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (num_samples, timesteps, features).
        
        Returns:
        - torch.Tensor: Weighted average tensor of shape (num_samples, 1, features).
        """
        # Apply softmax to ensure the weights sum to 1 and are positive
        softmax_weights = F.softmax(self.weights, dim=0)  # Shape (timesteps,)
        
        # Reshape weights to (1, timesteps, 1) to broadcast them across the batch and features
        weighted_input = x * softmax_weights.view(1, -1, 1)
        
        # Sum across the timesteps to get a weighted average
        weighted_average = torch.sum(weighted_input, dim=1, keepdim=True)  # Shape (num_samples, 1, features)
        
        return weighted_average