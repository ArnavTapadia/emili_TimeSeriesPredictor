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
from tensorflow.keras.regularizers import l1, l2
#hyperparam optimization imports
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import PredefinedSplit
from sklearn.metrics import make_scorer
from scikeras.wrappers import KerasRegressor
#%% LSTM model class definition
class LSTMEmotionPredictor:

    #loss and accuracy functions
    
    @staticmethod
    def custom_mse(y_true,y_pred):
        '''
        standard mse - euclidean distance between each prob vector, 
        then average across all samples and all timesteps equally weighted
        '''
        assert len(y_true.shape) == 3 and len(y_pred.shape) == 3
        # Cast tensors to the same type
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)



        # Calculate MSE
        squared_diff = tf.square(y_pred - y_true) #squared difference for each 7 vector at each timestamp
        euclidean_distances = tf.reduce_sum(squared_diff, axis=-1) #euclidean distance squared (mse) for each timestamp
        mean_distance_per_sample = tf.reduce_mean(euclidean_distances, axis=1) #average mse for each sample

        avg_euclidean_distance = tf.reduce_mean(mean_distance_per_sample, axis=0)

        return avg_euclidean_distance

    @staticmethod
    def custom_mse_time(y_true,y_pred):
        '''
        squaring over time with mse (euclidean distance) between each prob vector at each timestamp,

        then average across all samples and all timesteps equally weighted
        '''
        assert len(y_true.shape) == 3 and len(y_pred.shape) == 3
        # Cast tensors to the same type
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        # Calculate MSE
        squared_diff = tf.square(y_pred - y_true) #squared difference for each 7 vector at each timestamp
        euclidean_distances = tf.reduce_sum(squared_diff, axis=-1) #euclidean distance squared (mse) for each timestamp
        mean_distance_per_sample = tf.reduce_sum(tf.square(euclidean_distances), axis=1) #mse of euclidean distances over time per sample

        avg_euclidean_distance = tf.reduce_mean(mean_distance_per_sample, axis=0) #avg across samples

        return avg_euclidean_distance
    
    def custom_KLDiv_mse(y_true,y_pred):
        '''
        calculate KL divergence of each prob vector
        take square and sum for the sample (across timestamps)
        average over the number of samples
        '''
        assert len(y_true.shape) == 3 and len(y_pred.shape) == 3
        # Cast tensors to the same type
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        epsilon = 1e-7

        # Compute KL divergence for each timestamp
        kl_div_time = tf.reduce_sum(y_true * (tf.math.log(y_true + epsilon) - tf.math.log(y_pred + epsilon)), axis=-1)
        #square kl div of each timestamp and sum for each sample over all readings in the sample
        mse_kl_div = tf.reduce_sum(tf.square(kl_div_time), axis = 1)

        return tf.reduce_mean(mse_kl_div) #average across samples
    
    @staticmethod
    def kl_divergence_loss(y_true, y_pred): #only really used for bayesian optimization since regular LSTM uses normal keras function
        assert len(y_true.shape) == 3 and len(y_pred.shape) == 3
        # Cast tensors to the same type
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        # Add a small epsilon to avoid division by zero or log of zero
        epsilon = 1e-7

        # Compute KL divergence for each timestamp
        kl_div = tf.reduce_sum(y_true * (tf.math.log(y_true + epsilon) - tf.math.log(y_pred + epsilon)), axis=-1)

        # Average over all timestamps and batches
        return tf.reduce_mean(kl_div)

    def __init__(self, input_shape, LSTMUnits = 64, lossFunc = custom_mse_time, nAddLSTMLayers=0, 
                 nTimeDistributedLayers=0, nIntermediateDenseUnits=32):
        """
        Initialize the LSTM model.

        Parameters:
        - input_shape (tuple): Shape of the input data (excluding batch size).
        """

        self.input_shape = input_shape
        self.LSTMUnits = LSTMUnits
        self.lossFunc = lossFunc
        self.baseModel = self.create_model(nAddLSTMLayers=nAddLSTMLayers, 
                 nTimeDistributedLayers=nTimeDistributedLayers, nIntermediateDenseUnits=nIntermediateDenseUnits)
        self.baseHistory = None
        
        self.optimizedModel = None
        self.optimizedParams = None
        self.optimizedHistory = None


    def create_model(self, lstm_units=None, learning_rate=0.001, nAddLSTMLayers=0, 
                 nTimeDistributedLayers=0, nIntermediateDenseUnits=32, AddTimeDistributedActivation = 'relu'):
        """
        Build the LSTM model with configurable layers.

        Parameters:
        - lstm_units: Number of units in each LSTM layer. If None, uses self.LSTMUnits.
        - learning_rate: Learning rate for the Adam optimizer.
        - nLSTMLayers: Number of LSTM layers to add.
        - nTimeDistributedLayers: Number of intermediate TimeDistributed Dense layers.
        - nIntermediateDenseUnits: Number of units in each intermediate Dense layer.

        Returns:
        - tf.keras.Model: Compiled LSTM model.
        """

        if lstm_units is None:
            lstm_units = self.LSTMUnits

        model = Sequential()
        model.add(LSTM(lstm_units, input_shape=self.input_shape, return_sequences=True))
        # Add LSTM layers
        for _ in range(nAddLSTMLayers):
                model.add(LSTM(lstm_units, return_sequences=True))

        # Add intermediate TimeDistributed Dense layers
        for _ in range(nTimeDistributedLayers):
            model.add(TimeDistributed(Dense(nIntermediateDenseUnits, activation=AddTimeDistributedActivation)))
        # model.add(TimeDistributed(Dense(1, activation = 'relu'))) #compressing all lstm_units hidden states into a 600x1
        model.add(Dense(300)) #fully connected layer
        # Add final TimeDistributed Dense layer
        model.add(TimeDistributed(Dense(7, activation='softmax')))

        # Compile the model
        model.compile(loss=self.lossFunc,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=[LSTMEmotionPredictor.kl_divergence_loss])
        
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
            history = self.baseModel.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                     validation_data=(x_val, y_val))
        else:
            history = self.baseModel.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

        return history

    @staticmethod
    def evaluate(model, x_test, y_test):
        """
        Evaluate the LSTM model on test data.

        Parameters:
        - x_test (np.ndarray): Test data.
        - y_test (np.ndarray): Test labels.

        Returns:
        - list: Evaluation results [loss, accuracy].
        """
        return model.evaluate(x_test, y_test)
    
    #bayesian optimization
    def hyperparamOptimize(self, filterMethod, x_train, y_train, x_val, y_val, n_iter = 5):
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
            'epochs': Integer(10, 25),
            'model__lstm_units': Integer(64, 128),
            'model__learning_rate': Real(1e-4, 1e-2, prior='log-uniform'),
            'model__nAddLSTMLayers': Integer(0, 2),
            'model__nTimeDistributedLayers': Integer(0, 3),
            'model__AddTimeDistributedActivation': Categorical(['relu', 'tanh', 'sigmoid']),
            'model__nIntermediateDenseUnits': Integer(16, 64)
        }

        #defining custom loss functions to work with flattened data
        reshapeLambda = lambda y: tf.reshape(y, (-1, y_train.shape[1], y_train.shape[2]))
        flattenedLossFunction = lambda y_true, y_pred: LSTMEmotionPredictor.kl_divergence_loss(reshapeLambda(y_true), reshapeLambda(y_pred))


        def build_model_flattened_data(input_shape, lstm_units=64, learning_rate=0.001, nAddLSTMLayers=0, 
                 nTimeDistributedLayers=1, nIntermediateDenseUnits=32, AddTimeDistributedActivation = 'relu'):
            

            flattened_input_shape = np.prod(input_shape)
            flattened_output_shape = input_shape[0] * input_shape[1]  # Assuming 7 emotion categories

            #step 1 is the unflatten the input. then predict. then reflatten the output

            model = Sequential()
            #add reshape layer from 2d to 3d
            model.add(tf.keras.layers.Reshape(input_shape, input_shape=(flattened_input_shape,)))

            #add first LSTM layer
            model.add(LSTM(lstm_units, input_shape=input_shape, return_sequences=True))

            # Add intermediary LSTM layers
            for _ in range(nAddLSTMLayers):
                    model.add(LSTM(lstm_units, return_sequences=True))

            # Add intermediate TimeDistributed Dense layers
            for _ in range(nTimeDistributedLayers):
                model.add(TimeDistributed(Dense(nIntermediateDenseUnits, activation=AddTimeDistributedActivation)))

            # Add final TimeDistributed Dense layer
            model.add(TimeDistributed(Dense(7, activation='softmax')))

            #add final reshape back to 2d
            model.add(tf.keras.layers.Reshape((flattened_output_shape,)))

            #defining loss function            
            model.compile(loss=flattenedLossFunction,
                    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    metrics=[flattenedLossFunction])
            
            return model


        # Create the BayesSearchCV object
        #build_fn is the bayesian optimizer build function
        bayes_search = BayesSearchCV(
            estimator=KerasRegressor(
                model=build_model_flattened_data,
                input_shape=(x_train.shape[1], x_train.shape[2]),  # Set the input shape
                lstm_units=64,
                learning_rate=0.001,
                epochs=10,
                batch_size=32,
                verbose=1
            ),
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=ps,
            n_jobs=1,
            verbose=2,
            scoring = make_scorer(score_func=(lambda yTrue, yPred: flattenedLossFunction(yTrue,yPred).numpy()), greater_is_better=False) #'neg_mean_squared_error' but works with flattened timestamp data
        ) #find best scoring method (None => scoring method of the estimator)

        # Fit the BayesSearchCV object
        bayes_search.fit(X, Y)

        # Get the best parameters and model
        self.optimizedParams = bayes_search.best_params_
        self.optimizedModel = self.create_model(lstm_units=self.optimizedParams['model__lstm_units'], 
                                    learning_rate=self.optimizedParams['model__learning_rate'],
                                    nAddLSTMLayers=self.optimizedParams['model__nAddLSTMLayers'], 
                                    nTimeDistributedLayers=self.optimizedParams['model__nTimeDistributedLayers'], 
                                    nIntermediateDenseUnits=self.optimizedParams['model__nIntermediateDenseUnits'],
                                    AddTimeDistributedActivation=self.optimizedParams['model__AddTimeDistributedActivation'])
        
        # Train the best model with the original 3D data using the train method
        self.optimizedHistory = self.train(x_train, y_train, 
                            epochs=self.optimizedParams['epochs'], 
                            batch_size=self.optimizedParams['batch_size'], 
                            validation_data=(x_val, y_val))

        # Evaluate the best model
        val_loss, val_accuracy = LSTMEmotionPredictor.evaluate(self.optimizedModel, x_val, y_val)

        

        return {
            'filter_method': filterMethod,
            'best_model': self.optimizedModel,
            'best_params': self.optimizedParams,
            'best_val_accuracy': val_accuracy,
            'history': self.optimizedHistory
        }






#%% testing out the functions - initializing
extractor = emotionFeatureExtractor()

# XData,YData = extractor.prepare_and_segment_data()
# and input_shape is defined based on your data shape


#%% Testing different resampling methods
#load data:
# filterMethods = ['ewma', 'interpolation', 'ewmainterp', 'interp_ewmaSmooth']#,'binnedewma', 'times_scores']
filterMethods = ['ewmainterp']
modelMap = {} #filterMethod:(model, history)
dataSplitMap = {}
optimizedMap = {} #filterMethod:{filterMethod,best_model,best_params,best_val_accuracy,history}
time_series_predictorPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for filterChoice in filterMethods:

    #creating training and testing split
    # xTr,yTr,xVal,yVal,xTest,yTest = extractor.load_from_data_saves(resample_method=filterChoice) #if you want to load from pre-saved and split data
    xTr,yTr,xVal,yVal,xTest,yTest = extractor.get_data_split(resample_method=filterChoice, random_state=5) #if you want to load realtime
    
    
    dataSplitMap[filterChoice] = {'xTr':xTr,
                                  'yTr':yTr,
                                  'xVal':xVal,
                                  'yVal':yVal,
                                  'xTest':xTest,
                                  'yTest':yTest}
    # Create an instance of LSTMEmotionPredictor
    input_shape = (xTr.shape[1], xTr.shape[2])  # Assuming xTr is 3D with shape (#minute long segments, #time steps, #features = 7)
    lstm_model = LSTMEmotionPredictor(input_shape, nAddLSTMLayers=0,  nTimeDistributedLayers=0, lossFunc=LSTMEmotionPredictor.kl_divergence_loss)

    # Train the LSTM model
    lstm_model.baseHistory = lstm_model.train(xTr, yTr, epochs=10, batch_size=32, validation_data=(xVal, yVal)) #TODO: Find better way to set

    #Finding optimized Model:
    # lstm_model.hyperparamOptimize(filterChoice, xTr, yTr, xVal, yVal, n_iter=1) #don't care about returns

    modelMap[filterChoice] = lstm_model

#%% plotting loss functions
# Set up the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
fig.suptitle('Model Performance Comparison vs Training Epochs', fontsize=16)

# Define colors for each model
num_models = len(filterMethods)
colors = plt.cm.rainbow(np.linspace(0, 1, num_models))
color_dict = {method: colors[i] for i, method in enumerate(filterMethods)}

# Function to plot models
for filterMethod, model in modelMap.items():
    color = color_dict[filterMethod]
    
    # Plot training loss for unoptimized model
    ax1.plot(model.baseHistory.history['loss'], color=color, 
            label=f'{filterMethod} (train)', 
            linewidth=2, linestyle='-')
    
    # Plot validation loss
    ax1.plot(model.baseHistory.history['val_loss'], color=color, 
            label=f'{filterMethod} (val)', 
            linewidth=2, linestyle='--')
    

    #do same plots for optimized model (if it exists)
    if model.optimizedModel:
        ax2.plot(model.optimizedHistory.history['loss'], color=color, 
                label=f'{filterMethod} (train)', 
                linewidth=2, linestyle='-')
        
        ax2.plot(model.optimizedHistory.history['val_loss'], color=color, 
                label=f'{filterMethod} (val)', 
                linewidth=2, linestyle='--')
       
    
#making graphs look nice
ax1.set_title('Base Models: Training and Validation Loss', fontsize=14)
ax2.set_title('Optimized Models: Training and Validation Loss', fontsize=14)

for ax in [ax1, ax2]:
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_yscale('log')  # Set y-axis to logarithmic scale for better visibility

# Set common x-label
fig.text(0.5, 0.04, 'Epoch', ha='center', fontsize=12)

# Adjust layout and display
plt.tight_layout()
plt.show()



#%% comparing yVal for specific iSample
%matplotlib widget
iSample = np.random.randint(0, xVal.shape[0])  # choose random sample
num_features = 7

# Define colors for each model (base and optimized)
num_colors = len(filterMethods) * 2  # For both base and optimized models
colors = plt.cm.rainbow(np.linspace(0, 1, num_colors))

fig, axes = plt.subplots(num_features, 1, figsize=(12, 10), sharex=True, sharey=True)
fig.suptitle(f'yPred Features Prediction vs Time for Different Models, iSample {iSample}')

# True prediction (taking for 1st filterMethod as baseline)
xTrue_features = dataSplitMap[filterMethods[0]]['xVal'][iSample]
x_time = np.arange(0,0.1 * np.shape(xTrue_features)[0], 0.1)

yTrue_features = dataSplitMap[filterMethods[0]]['yVal'][iSample]
y_time = np.arange(0.1 * np.shape(yTrue_features)[0],0.2 * np.shape(yTrue_features)[0], 0.1)

# Plot true baseline
for i in range(num_features):
    #plotting xTrue
    axes[i].plot(x_time, xTrue_features[:, i], label='xTrue', color='black', linestyle='-', marker='o', markersize=3)
    axes[i].plot(y_time, yTrue_features[:, i], label='yTrue', color='black', linestyle='-', marker='o', markersize=3)

for idx, (filterMethod, data) in enumerate(dataSplitMap.items()):
    resampled_xVal = np.array([data['xVal'][iSample]])

    # Base model
    model = modelMap[filterMethod]
    yPred_features = model.baseModel.predict(resampled_xVal)

    for i in range(num_features):
        axes[i].plot(y_time, yPred_features[0, :, i], label=filterMethod, color=colors[idx*2], linestyle='-', marker='o', markersize=2)

    # Optimized model
    if model.optimizedModel:
        opt_yPred_features = model.optimizedModel.predict(resampled_xVal)

        for i in range(num_features):
            axes[i].plot(y_time, opt_yPred_features[0, :, i], label=f'optimized_{filterMethod}', color=colors[idx*2+1], linestyle='-', marker='o', markersize=2)

# Label subplots
for i in range(num_features):
    axes[i].set_ylabel(f'Feature {i + 1}')
    axes[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[i].grid(True)

axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.subplots_adjust(top=0.95, right=0.8)  # Adjust title position and make room for legend
axes[-1].set_xlim(40, 120)
axes[-1].set_ylim(0, 1)
plt.show()

# Print model summaries
print('\nBaseline Model')
model.baseModel.summary()

if model.optimizedModel:
    print('\nOptimized Model')
    model.optimizedModel.summary()

#%%
#evaluate loss on training and val for the models 
for method, model in modelMap.items():
    loss, accuracy = LSTMEmotionPredictor.evaluate(model.baseModel, dataSplitMap[method]['xTr'], dataSplitMap[method]['yTr'])
    print(f'{method}:')
    print(f'  Training loss: {loss:.4f}')
    loss, accuracy = LSTMEmotionPredictor.evaluate(model.baseModel, dataSplitMap[method]['xVal'], dataSplitMap[method]['yVal'])
    print(f'  Val loss: {loss:.4f}')

# %% Compare to model that predicts only the mean

#obtaining the test data
xVal = dataSplitMap['ewmainterp']['xVal']
yVal = dataSplitMap['ewmainterp']['xVal']

def meanPredictor(x):
    #x is the 3dimensional test array
    #this function predicts the mean vector probability vector given the previous one minute of data
    meanPerSample = np.mean(x, axis = 1, keepdims = True) #taking the mean over the timestamps for each sample

    return np.tile(meanPerSample, (1,600,1))

def geometricmeanPredictor(x):
    #x is the 3dimensional test array
    # Calculate the geometric mean along the axis 1 (timestamps)
    # To avoid log of zero, we use np.where to set small values to a tiny positive value
    epsilon = 1e-10
    x = np.where(x > 0, x, epsilon)
    
    # Compute the geometric mean along axis 1
    log_x = np.log(x)
    mean_log_x = np.mean(log_x, axis=1, keepdims=True)
    geometric_mean = np.exp(mean_log_x)
    
    # Tile the mean across all timestamps
    result = np.tile(geometric_mean, (1, x.shape[1], 1))
    
    return result

def mse_acrossTimestamps(y_true, y_pred): #only really used for bayesian optimization since regular LSTM uses normal keras function
    assert len(y_true.shape) == 3 and len(y_pred.shape) == 3

    # Cast tensors to the same type
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Calculate MSE
    squared_diff = tf.square(y_pred - y_true) #squared difference for each 7 vector at each timestamp
    euclidean_distances = tf.reduce_sum(squared_diff, axis=-1) #euclidean distance squared (mse) for each timestamp

    return tf.reduce_mean(euclidean_distances, axis = 0)


model = modelMap['ewmainterp']

print('\nKL Loss averaged across all timesteps in validation set')
print('Mean model KL Loss', LSTMEmotionPredictor.kl_divergence_loss(yVal,tf.convert_to_tensor(meanPredictor(xVal))).numpy())
print('Geometric Mean model KL Loss', LSTMEmotionPredictor.kl_divergence_loss(yVal,tf.convert_to_tensor(geometricmeanPredictor(xVal))).numpy())

print('Base model KL Loss', LSTMEmotionPredictor.kl_divergence_loss(yVal, model.baseModel.predict(xVal)).numpy())
if model.optimizedModel:
    print('Optimized model KL Loss', LSTMEmotionPredictor.kl_divergence_loss(yVal, model.optimizedModel.predict(xVal)).numpy())

print('\n\nMSE loss first averaged over time then samples')
print('Mean model mse Loss', LSTMEmotionPredictor.custom_mse(yVal,tf.convert_to_tensor(meanPredictor(xVal))).numpy())
print('Geometric Mean model mse Loss', LSTMEmotionPredictor.custom_mse(yVal,tf.convert_to_tensor(geometricmeanPredictor(xVal))).numpy())
print('Base model mse Loss', LSTMEmotionPredictor.custom_mse(yVal, model.baseModel.predict(xVal)).numpy())
if model.optimizedModel:
    print('Optimized model mse Loss', LSTMEmotionPredictor.custom_mse(yVal, model.optimizedModel.predict(xVal)).numpy())


# %% plot KL_divergence vs time averaged over test points
    
def kl_div_for_timesteps(y_true, y_pred): #only really used for bayesian optimization since regular LSTM uses normal keras function
    assert len(y_true.shape) == 3 and len(y_pred.shape) == 3

    # Cast tensors to the same type
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Add a small epsilon to avoid division by zero or log of zero
    epsilon = 1e-7

    # Compute KL divergence for each timestamp
    kl_div = tf.reduce_sum(y_true * (tf.math.log(y_true + epsilon) - tf.math.log(y_pred + epsilon)), axis=-1)

    # Average over all samples
    return tf.reduce_mean(kl_div, axis = 0) #should return a 600x1 vector

meanPredictorLoss = kl_div_for_timesteps(yVal, tf.convert_to_tensor(meanPredictor(xVal)))
geometricmeanPredictorLoss = kl_div_for_timesteps(yVal, tf.convert_to_tensor(geometricmeanPredictor(xVal))) 

baseModelLoss = kl_div_for_timesteps(yVal, model.baseModel.predict(xVal))
if model.optimizedModel:
    optimizedModelLoss = kl_div_for_timesteps(yVal, model.optimizedModel.predict(xVal))


fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
fig.suptitle('KL Divergence vs Time Averaged Across Samples in Validation Set')

axes.plot(y_time, meanPredictorLoss, label='Mean Predictor', color='black', linestyle='-', marker='o', markersize=3)
axes.plot(y_time, geometricmeanPredictorLoss, label='Geometric Mean Predictor', color='green', linestyle='-', marker='o', markersize=3)
axes.plot(y_time, baseModelLoss, label='Base Predictor (Trained with kl divergence)', color='red', linestyle='-', marker='o', markersize=3)
if model.optimizedModel:
    axes.plot(y_time, optimizedModelLoss, label='Optimized Predictor (Trained with kl divergence)', color='blue', linestyle='-', marker='o', markersize=3)

axes.set_xlabel('Time')
axes.set_ylabel('KL Loss')
plt.tight_layout()
axes.grid(True)
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.subplots_adjust(top=0.95, right=0.8)  # Adjust title position and make room for legend
axes.set_xlim(60, 120)
plt.show()
# %%
