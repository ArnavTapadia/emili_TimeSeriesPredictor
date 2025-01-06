import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedAverageModel(nn.Module):
    def __init__(self, timesteps, features):
        """
        Weighted Average Model where the weights across timesteps are learned.
        Essentially a multilayer perceptron/linear neural network

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
    def __init__(self, timesteps, features, hidden_units=64, forecast_length=1, activation = 'linear'):
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
        self.activation = activation

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
    

class LSTMMultivariate(nn.Module):
    def __init__(self, timesteps, features, lstm_units=64, hidden_fc_layers=0, hidden_units=128, forecast_length=1, nn_activation='linear'):
        """
        LSTM Model for univariate time series with multi-step one-shot prediction.

        Parameters:
        - timesteps (int): Number of timesteps in the input sequence.
        - features (int): Number of features at each timestep.
        - lstm_units (int): Number of units in the LSTM layer.
        - forecast_length (int): Number of steps to predict into the future (in one shot).
        - hidden_fc_layers (int): Number of Fully Connected hidden NN layers after the LSTM layer
        - hidden_units (int): Number of nodes in the hidden layers
        - nn_activation (string): Activation to use on the last fully connected output layer
        """
        super(LSTMMultivariate, self).__init__()

        self.timesteps = timesteps
        self.features = features
        self.lstm_units = lstm_units
        self.hidden_units = hidden_units
        assert hidden_fc_layers == 0 or hidden_fc_layers < 3  # only supports 0 or 1 right now
        self.hidden_fc_layers = hidden_fc_layers
        assert nn_activation == 'relu' or nn_activation == 'linear'  # only supports linear or relu now
        self.nn_activation = nn_activation
        self.forecast_length = forecast_length  # Number of timesteps to predict

        # LSTM layer: input_size is features, hidden_size is lstm_units
        self.lstm = nn.LSTM(input_size=features, hidden_size=lstm_units, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        if hidden_fc_layers > 0:
            self.output = nn.Linear(hidden_units, forecast_length * features)
        else:
            self.output = nn.Linear(lstm_units, forecast_length * features)

    def forward(self, x, return_activations=False):
        """
        Forward pass of the model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, timesteps, features).
        - return_activations (bool): If True, return hidden state activations.

        Returns:
        - torch.Tensor: Predicted output for the next `forecast_length` timesteps,
                        shape (batch_size, forecast_length, features).
        - activations (optional): If `return_activations` is True, returns hidden state activations of shape 
                                  (batch_size, timesteps, lstm_units).
        """
        # Pass input through LSTM layer
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, timesteps, lstm_units)

        # If requested, return activations (i.e., hidden states for each LSTM unit across timesteps)
        if return_activations:
            return lstm_out

        # Use the output from the last timestep of the LSTM
        last_timestep_output = lstm_out[:, -1, :]  # shape: (batch_size, lstm_units)

        if self.hidden_fc_layers == 1:
            hidden_out = F.relu(self.fc1(last_timestep_output))
        elif self.hidden_fc_layers == 2:
            hidden_out1 = F.relu(self.fc1(last_timestep_output))
            hidden_out = F.relu(self.fc2(hidden_out1))
        else:
            hidden_out = last_timestep_output

        # Pass through the fully connected layer to predict the full forecast
        if self.nn_activation == 'relu':
            fc_output = F.relu(self.output(hidden_out))  # shape: (batch_size, forecast_length * features)
        else:
            fc_output = self.output(hidden_out)

        # Reshape the output to be (batch_size, forecast_length, features)
        output = fc_output.view(-1, self.forecast_length, self.features)

        return output  # shape: (batch_size, forecast_length, features)