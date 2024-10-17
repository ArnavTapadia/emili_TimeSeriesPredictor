import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

'''
Functions for training and inference of the models
'''

def train(model, xTrain, yTrain, xVal=None, yVal=None, epochs=10, batch_size=32, lr=0.001, shuffle=False):
    """
    General training function for PyTorch models.
    
    Parameters:
    - model (nn.Module): The PyTorch model to be trained.
    - xTrain (numpy.ndarray): Training data of shape (num_samples, timesteps, features).
    - yTrain (numpy.ndarray): Training labels of shape (num_samples, target_dim).
    - xVal (numpy.ndarray, optional): Validation data of shape (num_samples, timesteps, features).
    - yVal (numpy.ndarray, optional): Validation labels of shape (num_samples, target_dim).
    - epochs (int, optional): Number of epochs to train (default is 10).
    - batch_size (int, optional): Batch size (default is 32).
    - lr (float, optional): Learning rate for the optimizer (default is 0.001).
    - device (str, optional): Device to use for training ('cpu' or 'cuda'). If None, defaults to 'cpu'.
    
    Returns:
    - training_loss_history (list): History of training loss per epoch.
    - validation_loss_history (list): History of validation loss per epoch (if validation data is provided).
    """

    # Convert numpy arrays to PyTorch tensors and move them to the device
    xTrain = torch.from_numpy(xTrain).float()
    yTrain = torch.from_numpy(yTrain).float()
    
    if xVal is not None and yVal is not None:
        xVal = torch.from_numpy(xVal).float()
        yVal = torch.from_numpy(yVal).float()

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Can be changed based on the task (e.g., CrossEntropyLoss for classification)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # History to store loss per epoch
    training_loss_history = []
    validation_loss_history = []

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        epoch_loss = 0.0
        
        if shuffle:
            # Shuffle the data at the beginning of each epoch
            permutation = torch.randperm(xTrain.size(0))
        else:
            # If no shuffling, maintain original order
            permutation = torch.arange(xTrain.size(0))

        # Process each batch
        for i in range(0, len(xTrain), batch_size):
            indices = permutation[i:i+batch_size]
            x_batch = xTrain[indices]
            y_batch = yTrain[indices]
            
            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            outputs = model(x_batch)
            assert(outputs.shape == y_batch.shape) #ensuring input and output sizes always match
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        # Average loss for the epoch
        training_loss_history.append(epoch_loss / len(xTrain))

        # Validation step (if provided)
        if xVal is not None and yVal is not None:
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            with torch.no_grad():  # No need to track gradients during validation
                val_outputs = model(xVal)
                val_loss = criterion(val_outputs, yVal).item()

            validation_loss_history.append(val_loss)
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(xTrain):.4f}, Validation Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss/len(xTrain):.4f}')

    if xVal is not None and yVal is not None:
        return training_loss_history, validation_loss_history
    else:
        return training_loss_history


def predict(model, input_data):
    """
    Makes predictions using the provided PyTorch model and input data.
    
    Parameters:
    - model (nn.Module): The PyTorch model used for predictions.
    - input_data (numpy.ndarray): Input data of shape (batch_size, timesteps, features).
    
    Returns:
    - output (numpy.ndarray): Model predictions.
    """
    model.eval()  # Set the model to evaluation mode
    input_tensor = torch.from_numpy(input_data).float()  # Convert numpy to PyTorch tensor

    with torch.no_grad():  # No need to calculate gradients for predictions
        output_tensor = model(input_tensor)

    return output_tensor.numpy()  # Convert back to numpy array

def recursive_predict(model, input_data, nRecursive_predictions):
    """
    Perform recursive prediction with a PyTorch model by using the last predicted value as the next input.

    Parameters:
    - model (nn.Module): The PyTorch model used for predictions.
    - input_data (numpy.ndarray): Initial input data of shape (batch_size, timesteps, features).
    - nRecursive_predictions (int): Number of recursive predictions to make.
    
    Returns:
    - predictions (numpy.ndarray): Array of recursive predictions.
    """
    predictions = []
    current_input = torch.from_numpy(input_data).float()  # Convert to tensor

    model.eval()
    
    for _ in range(nRecursive_predictions):
        with torch.no_grad():
            # Predict the next timestep
            next_pred = model(current_input)
            predictions.append(next_pred)  # Store prediction

        # Shift the input sequence up by 1
        current_input = current_input.roll(shifts=-next_pred.shape[1], dims=1)
        # Replace the last timestep with the new prediction
        current_input[:, -next_pred.shape[1]:, :] = next_pred

    return torch.cat(predictions, dim=1)

def calculate_mse_loss(predictions, true_values):
    """
    Calculates the MSE loss over a set of predictions and true values.

    Parameters:
    - predictions: Predicted values (numpy array or PyTorch tensor) of shape (n_samples, timesteps, features).
    - true_values: True values (numpy array or PyTorch tensor) of the same shape as predictions.

    Returns:
    - mse_loss: The MSE loss calculated over all samples.
    """
    # Convert inputs to PyTorch tensors if they are not already
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions).float()
    if isinstance(true_values, np.ndarray):
        true_values = torch.from_numpy(true_values).float()
    
    # Ensure predictions and true_values are of the same shape
    assert predictions.shape == true_values.shape, "Predictions and true values must have the same shape."
    
    # Calculate MSE loss using PyTorch's MSELoss
    criterion = nn.MSELoss()
    mse_loss = criterion(predictions, true_values).item()

    return mse_loss