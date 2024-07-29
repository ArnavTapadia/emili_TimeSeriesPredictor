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
    
    #build model function just for use for bayesian optimization (keras regressor only works on 2d data)
    def bayes_opt_build_model(self, lstm_units=64, learning_rate=0.001):
        flattened_input_shape = np.prod(self.input_shape)
        flattened_output_shape = self.input_shape[0] * 7  # Assuming 7 emotion categories

        model = Sequential([
            tf.keras.layers.Reshape(self.input_shape, input_shape=(flattened_input_shape,)),
            LSTM(lstm_units, return_sequences=True),
            TimeDistributed(Dense(7, activation='softmax')),
            tf.keras.layers.Reshape((flattened_output_shape,))
        ])
        
        model.compile(loss=self.lossFunc,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=['accuracy'])
        return model

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
        #flattening so data is 2d (#samples, 600*7)
        X_train_flat = x_train.reshape(x_train.shape[0], -1)
        X_val_flat = x_val.reshape(x_val.shape[0], -1)
        X = np.concatenate((X_train_flat, X_val_flat), axis=0)
        # Flatten the predictions also
        Y_train_flat = y_train.reshape(y_train.shape[0], -1)
        Y_val_flat = y_val.reshape(y_val.shape[0], -1)
        Y = np.concatenate((Y_train_flat, Y_val_flat), axis=0)

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
        
        estimator=KerasRegressor(
                model=self.bayes_opt_build_model, epochs=10, batch_size=32,
                verbose=0
            )

        def build_model(input_shape, lstm_units=64, learning_rate=0.001):

            model = Sequential([
                LSTM(lstm_units, input_shape=self.input_shape, return_sequences=True),
                TimeDistributed(Dense(7, activation='softmax'))
            ])
            
            model.compile(loss='mean_squared_error',
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        metrics=['accuracy'])
            return model

        def build_model2(input_shape, lstm_units=64, learning_rate=0.001):
            flattened_input_shape = np.prod(input_shape)
            flattened_output_shape = input_shape[0] * 7  # Assuming 7 emotion categories

            #step 1 is the unflatten the input. then predict. then reflatten the output
            model = Sequential([
                tf.keras.layers.Reshape(input_shape, input_shape=(flattened_input_shape,)),
                LSTM(lstm_units, return_sequences=True),
                TimeDistributed(Dense(7, activation='softmax')),
                tf.keras.layers.Reshape((flattened_output_shape,))
            ])
            model.summary()


            def custom_mse(y_true,y_pred): #used for flattened data
                # Reshape predictions back to 3D
                y_pred_3d = tf.reshape(y_pred, (-1, y_train.shape[1], y_train.shape[2]))
                y_true_3d = tf.reshape(y_true, (-1, y_train.shape[1], y_train.shape[2]))

                # Calculate MSE
                return tf.reduce_mean(tf.square(y_pred_3d - y_true_3d))

            def custom_accuracy(y_true, y_pred):
                # Reshape predictions back to 3D
                y_pred_3d = tf.reshape(y_pred, (-1, y_train.shape[1], y_train.shape[2]))
                y_true_3d = tf.reshape(y_true, (-1, y_train.shape[1], y_train.shape[2]))
                
                # Calculate accuracy
                accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred_3d, axis=-1),
                                                        tf.argmax(y_true_3d, axis=-1)),
                                                tf.float32))
                return accuracy
            
            def cosine_similarity_accuracy(y_true, y_pred):
                # Reshape
                y_true = tf.reshape(y_true, (-1, y_train.shape[1], y_train.shape[2]))
                y_pred = tf.reshape(y_pred, (-1, y_train.shape[1], y_train.shape[2]))
                
                # Compute cosine similarity for each timestamp
                similarity = tf.reduce_sum(y_true * y_pred, axis=-1) / (
                    tf.norm(y_true, axis=-1) * tf.norm(y_pred, axis=-1) + 1e-7
                )
                
                # Average over all timestamps and batches
                return tf.reduce_mean(similarity)

            model.compile(loss=custom_mse,
                        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        metrics=[custom_accuracy, cosine_similarity_accuracy]) #TODO: Determine a better accuracy metric
            
            return model


        testingModel1 = build_model(input_shape=(x_train.shape[1], x_train.shape[2]),lstm_units=64, learning_rate=0.001)
                                   
        testingModel1.fit(x_train,y_train, epochs = 10, batch_size = 32)

        testingModel2 = build_model2(input_shape=(x_train.shape[1], x_train.shape[2]),lstm_units=64, learning_rate=0.001)
                                   
        model2 = testingModel2.fit(X_train_flat,Y_train_flat, epochs = 10, batch_size = 32)
        

        #testing code: ----
        flattened_input_shape = np.prod(input_shape)
        reshape_layer = tf.keras.layers.Reshape(input_shape, input_shape=(flattened_input_shape,))
        reshape_output = reshape_layer(X_train_flat)

        flattened_output_shape = input_shape[0] * 7  # Assuming 7 emotion categories

        output_reshape = tf.keras.layers.Reshape((flattened_output_shape,))
        output_reshape_output = output_reshape(y_train)
        #----


        model = KerasRegressor(
            build_fn=build_model,
            input_shape=(x_train.shape[1], x_train.shape[2]),  # Set the input shape
            lstm_units=64,
            learning_rate=0.001,
            epochs=10,
            batch_size=32,
            verbose=1
        )

        model.fit(x_train, y_train)






        estimator.fit(X, Y, epochs=5, batch_size=16)

        # Create the BayesSearchCV object
        #build_fn is the bayesian optimizer build function
        bayes_search = BayesSearchCV(
            estimator=KerasRegressor(
                model=self.bayes_opt_build_model,  epochs=10, batch_size=32,
                verbose=0
            ),
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
        self.model = self.create_model(lstm_units=best_params['lstm_units'], 
                                    learning_rate=best_params['learning_rate'])
        
        # Train the best model with the original 3D data using the train method
        history = self.train(x_train, y_train, 
                            epochs=best_params['epochs'], 
                            batch_size=best_params['batch_size'], 
                            validation_data=(x_val, y_val))

        # Evaluate the best model
        val_loss, val_accuracy = self.evaluate(x_val, y_val)

        return {
            'filter_method': filterMethod,
            'best_model': self.model,
            'best_params': best_params,
            'best_val_accuracy': val_accuracy,
            'history': history
        }


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
    # history = lstm_model.train(xTr, yTr, epochs=10, batch_size=32, validation_data=(xVal, yVal))

    # modelMap[testMethod] = (lstm_model,history)

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
