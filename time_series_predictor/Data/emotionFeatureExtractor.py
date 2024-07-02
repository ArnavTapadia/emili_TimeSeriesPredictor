import numpy as np
import pandas as pd
import json
import os


class emotionFeatureExtractor:
    
    def read_emotion_logs(log_dir = 'time_series_predictor/Data'):
        '''
        log_dir has n (= 100) time series of varying length between 1-10 minutes long
        each time series contains a log of the emotion_data with ~ 10 readings per second
        '''
        all_data = []
        for file_name in os.listdir(log_dir): #iterating through each log file
            if file_name.endswith('.json'):
                with open(os.path.join(log_dir, file_name), 'r') as file:
                    data = json.load(file)
                    all_data.append(data) #all_data should be of length n and each element is a full time series of emotion data
        return all_data
    
    def resample_data(file_data, target_freq='100L'): #resampling the data linearly so we have 10 readings per second
        #'100L' translates to 10 Hz frequency
        df = pd.DataFrame(file_data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        df = df.resample(target_freq).interpolate(method='linear')
        scores_array = np.stack(df['scores'].apply(lambda x: np.array(x)).values)

        return scores_array

    def prepare_data(all_data): #returns matrix of data
        max_length = max(len(file_data) for file_data in all_data) #takes the maximum log length
        num_files = len(all_data) #should be = #log files
        num_features = 7  # Number of emotion scores #length fo emotion vector

        # Initialize a 3D array with shape (num_files, max_length, num_features)
        data_array = np.zeros((num_files, max_length, num_features)) #should be 100xmax(logfilelength)x7

        for i, file_data in enumerate(all_data):
            df = pd.DataFrame(file_data)
            scores_array = np.array(df['scores'].tolist())
            data_array[i, :scores_array.shape[0], :] = scores_array

        return data_array
    
    

    def segmented_data(data_array, segment_length = 600,stride = 600):
        '''
        Method #1 for training model:
        Segments data so that only the previous 1 minute (600 readings) are used to make a guess for the next 1 minute
        data_array should be an np.matrix of size ~100xmax(logfilelength)x7 (100 = #time series')
        '''

        X = Y = []
        num_files, max_length, num_features = data_array.shape

        for nTimeSeries in range(num_files):
            file_data = data_array[nTimeSeries]
        for start in range(0, max_length - segment_length, stride):
            end = start + segment_length
            if end + segment_length <= max_length:
                X.append(file_data[start:end])
                Y.append(file_data[end:end + segment_length])
            #Note some of X and Y are going to be mostly 0's rather than vectors
            #X and Y should have size ~ # of minutes of data x600x7
        return np.array(X), np.array(Y)
    

    def train_val_testing_split(X,Y):
        raise NotImplementedError