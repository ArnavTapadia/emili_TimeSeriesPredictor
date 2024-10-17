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
    
class MultiStepFullyConnectedNN(nn.Module):
    def __init__(self, timesteps, features, hidden_units=64, forecast_length=1):
        """
        Fully Connected Neural Network for multi-step time series prediction.

        Parameters:
        - timesteps (int): Number of timesteps in the input sequence.
        - features (int): Number of features in each timestep.
        - hidden_units (int): Number of units in the hidden layer.
        - forecast_length (int): Number of steps to predict into the future.
        """
        super(MultiStepFullyConnectedNN, self).__init__()

        self.timesteps = timesteps
        self.features = features
        self.forecast_length = forecast_length

        # Calculate input size based on the number of timesteps and features
        self.input_size = timesteps * features

        # Define the layers
        self.fc1 = nn.Linear(self.input_size, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, forecast_length * features)

    def forward(self, x):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, timesteps, features).

        Returns:
        - torch.Tensor: Predicted output for the next `forecast_length` timesteps,
                        shape (batch_size, forecast_length, features).
        """
        # Flatten the input
        x = x.view(-1, self.input_size)

        # Forward pass through the network
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape to match the output shape (batch_size, forecast_length, features)
        output = x.view(-1, self.forecast_length, self.features)
        return output