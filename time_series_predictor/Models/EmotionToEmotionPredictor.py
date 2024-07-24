#%% imports
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) #going up several files until emili_TimeSeriesPredictor
from time_series_predictor.Data.emotionFeatureExtractor import emotionFeatureExtractor
#Modeling imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
#hyperparam optimization imports
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import PredefinedSplit
from scikeras.wrappers import KerasRegressor
#%% LSTM model class definition
class LSTMEmotionPredictor:
    def __init__(self, input_shape, LSTMUnits = 64, lossFunc = 'kl_divergence'):
        """
        Initialize the LSTM model.

        Parameters:
        - input_shape (tuple): Shape of the input data (excluding batch size).
        """

        self.input_shape = input_shape
        self.LSTMUnits = LSTMUnits
        self.lossFunc = lossFunc
        self.model = self.create_model()


    def create_model(self, lstm_units = None, learning_rate = 0.001):
        """
        Build the LSTM model.

        Parameters:
        - input_shape (tuple): Shape of the input data (excluding batch size).

        Returns:
        - tf.keras.Model: Compiled LSTM model.
        """

        if lstm_units is None:
            lstm_units = self.LSTMUnits

        model = Sequential([
            LSTM(lstm_units, input_shape=self.input_shape, return_sequences=True),
            TimeDistributed(Dense(7, activation='softmax'))
        ])
        # Compile the model
        model.compile(loss=self.lossFunc,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
    
    def hyperparamOptimize(self, filterMethod, x_train, y_train, x_val, y_val, n_iter = 50):
        """
        Perform Bayesian optimization for hyperparameter tuning of the LSTM model.

        Parameters:
        - x_train (np.ndarray): Training data
        - y_train (np.ndarray): Training labels
        - x_val (np.ndarray): Validation data
        - y_val (np.ndarray): Validation labels
        - filter_method (str): Name of the filter method being used
        - n_iter (int): Number of iterations for Bayesian optimization

        Returns:
        - dict: Best model, its hyperparameters, and performance
        """

        #combining train and val sets (split will be maintained in test_fold)
        X = np.concatenate((x_train, x_val), axis=0)
        Y = np.concatenate((y_train, y_val), axis=0)

        # Create a PredefinedSplit
        test_fold = np.concatenate((np.full(x_train.shape[0], -1), np.zeros(x_val.shape[0]))) #-1's for training 0 for val
        ps = PredefinedSplit(test_fold)

        # Define the search space
        search_spaces = {
            'batch_size': Categorical([16, 32, 64, 128]),
            'epochs': Integer(5, 100),
            'model__lstm_units': Integer(16, 256),
            'model__learning_rate': Real(1e-4, 1e-2, prior='log-uniform')
        }

        # Create the BayesSearchCV object
        bayes_search = BayesSearchCV(
            estimator=KerasRegressor(build_fn=self.create_model, verbose=0),
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=ps,
            n_jobs=1,
            verbose=2,
            scoring='neg_mean_squared_error'
        ) #TODO: test scoring method as 'accuracy'

        # Fit the BayesSearchCV object
        bayes_search.fit(X, Y)

        # Get the best parameters and model
        best_params = bayes_search.best_params_

        best_model = self.create_model(lstm_units=best_params['lstm_units'], learning_rate=best_params['model__learning_rate'])
        best_model.fit(x_train, y_train, 
                       epochs=best_params['epochs'], 
                       batch_size=best_params['batch_size'], 
                       validation_data=(x_val, y_val),
                       verbose=0)


        # Evaluate the best model
        val_loss, val_accuracy = best_model.evaluate(x_val, y_val, verbose=0)

        return {
            'filter_method': filterMethod,
            'best_model': best_model,
            'best_params': best_params,
            'best_val_accuracy': val_accuracy
            }

        # hyperParamMap = {} #maps each tuple combination of hyperparameters to 
        # return hyperParamMap

#%% testing out the functions - initializing
extractor = emotionFeatureExtractor()

# XData,YData = extractor.prepare_and_segment_data()
# and input_shape is defined based on your data shape


#%% Testing different resampling methods
#load data:
filterMethods = ['ewma', 'binnedewma', 'interpolation', 'ewmainterp', 'interp_ewmaSmooth']#, 'times_scores']
modelMap = {}
time_series_predictorPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for testMethod in filterMethods:
#to load data file name is os.path.join('time_series_predictor/Data/Data_Saves/Preprocessed', fName + '_x.npy') or y.npy
    XData = np.load(os.path.join(time_series_predictorPath,'Data/Data_Saves/Preprocessed', testMethod + '_x.npy'))
    YData = np.load(os.path.join(time_series_predictorPath,'Data/Data_Saves/Preprocessed', testMethod + '_y.npy'))


    #creating training and testing split
    xTr,yTr,xVal,yVal,xTest,yTest = extractor.train_val_testing_split(XData,YData, random_state=5)
    # Create an instance of LSTMEmotionPredictor
    input_shape = (xTr.shape[1], xTr.shape[2])  # Assuming xTr is 3D with shape (#minute long segments, #time steps, #features = 7)
    lstm_model = LSTMEmotionPredictor(input_shape)

    # Train the LSTM model
    history = lstm_model.train(xTr, yTr, epochs=10, batch_size=32, validation_data=(xVal, yVal))

    modelMap[testMethod] = (lstm_model,history)

    #Finding optimized Model:
    optimized = lstm_model.hyperparamOptimize(testMethod, xTr, yTr, xVal, yVal)

#%% plotting
# Set up the plot
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Model Performance Comparison', fontsize=16)

# Define colors for each model
colors = plt.cm.rainbow(np.linspace(0, 1, len(filterMethods)))

# Plot accuracy and loss for each model
for (method, (model, history)), color in zip(modelMap.items(), colors):
    # Plot training accuracy
    ax1.plot(history.history['accuracy'], color=color, label=method, linewidth=2)
    
    # Plot validation accuracy
    ax2.plot(history.history['val_accuracy'], color=color, label=method, linewidth=2)
    
    # Plot training loss
    ax3.plot(history.history['loss'], color=color, label=method, linewidth=2)
    
    # Plot validation loss
    ax4.plot(history.history['val_loss'], color=color, label=method, linewidth=2)

# Customize training accuracy subplot
ax1.set_title('Training Accuracy', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.7)

# Customize validation accuracy subplot
ax2.set_title('Validation Accuracy', fontsize=14)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.legend(loc='lower right', fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.7)

# Customize training loss subplot
ax3.set_title('Training Loss', fontsize=14)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(True, linestyle='--', alpha=0.7)

# Customize validation loss subplot
ax4.set_title('Validation Loss', fontsize=14)
ax4.set_ylabel('Loss', fontsize=12)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, linestyle='--', alpha=0.7)

# Adjust layout and display
plt.tight_layout()
plt.show()

#%% Evaluate the model on test data

# Print test loss and accuracy for each model
print("\nTest Results:")
for method, (model, history) in modelMap.items():
    loss, accuracy = model.evaluate(xTest, yTest)
    print(f'{method}:')
    print(f'  Test loss: {loss:.4f}')
    print(f'  Test accuracy: {accuracy:.4f}')
    print()
# %%
