import torch
import torch.nn as nn


'''
LSTMUnivariateRecursivePred: predicts a recursive_forecast_length timesteps into the future.
If recursive_forecast_length > 1, then the loss will be over the WHOLE forecast rather than the first prediction during training
    - in this case it takes a long time to train without having much improvement over a single future timestep predictor
'''

class LSTMUnivariateRecursivePred(nn.Module):
    def __init__(self, timesteps, features, lstm_units=64, recursive_forecast_length=1):
        """
        LSTM Model for univariate time series with multi-step or recursive prediction.

        Parameters:
        - timesteps (int): Number of timesteps in the input sequence.
        - features (int): Number of features at each timestep.
        - lstm_units (int): Number of units in the LSTM layer.
        - recursive_forecast_length (int): Number of steps to recursively predict into the future.
        """
        super(LSTMUnivariateRecursivePred, self).__init__()
        
        self.timesteps = timesteps
        self.features = features
        self.lstm_units = lstm_units
        self.recursive_forecast_length = recursive_forecast_length  # Number of timesteps to predict

        # Define the LSTM layer
        self.lstm = nn.LSTM(input_size=features, hidden_size=lstm_units, batch_first=True)

        # Define the fully connected layer to map LSTM outputs to the correct number of features
        self.fc = nn.Linear(lstm_units, features)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, timesteps, features).
        
        Returns:
        - torch.Tensor: Predicted output for the next `recursive_forecast_length` timesteps,
                        shape (batch_size, recursive_forecast_length, features).
        """
        
        if self.recursive_forecast_length == 1:
            # Predict only one step (standard LSTM behavior)
            lstm_out, _ = self.lstm(x)
            output = self.fc(lstm_out[:, -1, :])  # Output for the last timestep
            return output.unsqueeze(1)  # Ensure shape: (batch_size, 1, features)
        
        else:
            # Recursive prediction with access to the full original sequence
            outputs = []  # Store all predicted timesteps
            current_input = x  # Start with the full original input sequence
            
            # First, pass through the entire input to get the LSTM output for the final timestep
            lstm_out, (hn, cn) = self.lstm(current_input)
            output = self.fc(lstm_out[:, -1, :])  # First prediction based on the last LSTM output
            outputs.append(output.unsqueeze(1))  # Add the first prediction

            # Recursively predict the next timesteps
            for _ in range(1, self.recursive_forecast_length):
                # Concatenate the previous input with the new prediction along the timestep axis
                new_input = torch.cat([current_input[:, 1:, :], output.unsqueeze(1)], dim=1)
                
                # Use the concatenated input to make the next prediction
                lstm_out, (hn, cn) = self.lstm(new_input, (hn, cn))  # Use the previous hidden state (hn, cn)
                output = self.fc(lstm_out[:, -1, :])  # Predict the next output
                outputs.append(output.unsqueeze(1))  # Store the output

                # Update current_input to include the new prediction
                current_input = new_input

            # Concatenate all predictions along the timestep axis
            return torch.cat(outputs, dim=1)  # Shape: (batch_size, recursive_forecast_length, features)


'''
LSTMUnivariateMultiStep: predicts forecast_length timesteps into the future in one shot.
'''

class LSTMUnivariateMultiStep(nn.Module):
    def __init__(self, timesteps, features, lstm_units=64, forecast_length=1):
        """
        LSTM Model for univariate time series with multi-step one-shot prediction.

        Parameters:
        - timesteps (int): Number of timesteps in the input sequence.
        - features (int): Number of features at each timestep.
        - lstm_units (int): Number of units in the LSTM layer.
        - forecast_length (int): Number of steps to predict into the future (in one shot).
        """
        super(LSTMUnivariateMultiStep, self).__init__()

        self.timesteps = timesteps
        self.features = features
        self.lstm_units = lstm_units
        self.forecast_length = forecast_length  # Number of timesteps to predict

        # LSTM layer: input_size is features, hidden_size is lstm_units
        self.lstm = nn.LSTM(input_size=features, hidden_size=lstm_units, batch_first=True)

        # Fully connected layer: Maps LSTM output to future timestep predictions
        # Instead of just mapping to features, we now map to forecast_length * features
        self.fc = nn.Linear(lstm_units, forecast_length * features)

    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, timesteps, features).
        
        Returns:
        - torch.Tensor: Predicted output for the next `forecast_length` timesteps,
                        shape (batch_size, forecast_length, features).
        """
        # Pass input through LSTM layer
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, timesteps, lstm_units)

        # Use the output from the last timestep of the LSTM
        last_timestep_output = lstm_out[:, -1, :]  # shape: (batch_size, lstm_units)

        # Pass through the fully connected layer to predict the full forecast
        fc_output = self.fc(last_timestep_output)  # shape: (batch_size, forecast_length * features)

        # Reshape the output to be (batch_size, forecast_length, features)
        output = fc_output.view(-1, self.forecast_length, self.features)

        return output  # shape: (batch_size, forecast_length, features)
