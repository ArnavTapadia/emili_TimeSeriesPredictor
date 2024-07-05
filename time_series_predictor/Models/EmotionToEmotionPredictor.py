import numpy as np
import tensorflow as tf
from time_series_predictor.Data.emotionFeatureExtractor import emotionFeatureExtractor
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense

class LSTMEmotionPredictor:
    def __init__(self, input_shape):
        """
        Initialize the LSTM model.

        Parameters:
        - input_shape (tuple): Shape of the input data (excluding batch size).
        """
        self.model = self.build_model(input_shape)

    def build_model(self, input_shape):
        """
        Build the LSTM model.

        Parameters:
        - input_shape (tuple): Shape of the input data (excluding batch size).

        Returns:
        - tf.keras.Model: Compiled LSTM model.
        """
        model = Sequential()
        # LSTM layer with 64 units #to be validated
        model.add(LSTM(64, input_shape=input_shape))
        # Output layer with 7 units (assuming 7 output classes for emotions)
        model.add(Dense(7, activation='softmax'))
        
        # Compile the model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        """
        Train the LSTM model.

        Parameters:
        - x_train (np.ndarray): Training data.
        - y_train (np.ndarray): Training labels.
        - epochs (int): Number of epochs for training.
        - batch_size (int): Batch size for training.
        - validation_data (tuple): Validation data as (x_val, y_val). Default is None.
        """
        if validation_data:
            x_val, y_val = validation_data
            history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                     validation_data=(x_val, y_val))
        else:
            history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        
        return history

    def evaluate(self, x_test, y_test):
        """
        Evaluate the LSTM model on test data.

        Parameters:
        - x_test (np.ndarray): Test data.
        - y_test (np.ndarray): Test labels.

        Returns:
        - list: Evaluation results [loss, accuracy].
        """
        return self.model.evaluate(x_test, y_test)

# Example usage:
extractor = emotionFeatureExtractor()
XData,YData = extractor.prepare_and_segment_data()
xTr,yTr,xVal,yVal,xTest,yTest = extractor.train_val_testing_split(XData,YData, random_state=5)
# and input_shape is defined based on your data shape

# Create an instance of LSTMEmotionPredictor
input_shape = (xTr.shape[1], xTr.shape[2])  # Assuming xTr is 3D with shape (#minute long segments, #time steps, #features = 7)
lstm_model = LSTMEmotionPredictor(input_shape)

# Train the LSTM model
history = lstm_model.train(xTr, yTr, epochs=10, batch_size=32, validation_data=(xVal, yVal))

# Evaluate the model on test data
loss, accuracy = lstm_model.evaluate(xTest, yTest)

print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
