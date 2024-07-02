import numpy as np
import pandas as pd
import json
import os


class emotionFeatureExtractor:
    def __init__(self, log_dir = 'time_series_predictor/Data', segment_length=600, stride=600, target_freq='100L'):
        self.log_dir = log_dir
        self.segment_length = segment_length
        self.stride = stride
        self.target_freq = target_freq


    def read_emotion_logs(self):
        '''
        log_dir has n (= 100) time series of varying length between 1-10 minutes long
        each time series contains a log of the emotion_data with ~ 10 readings per second
        '''
        all_data = []
        for file_name in os.listdir(self.log_dir): #iterating through each log file
            if file_name.endswith('.json'):
                with open(os.path.join(self.log_dir, file_name), 'r') as file:
                    data = json.load(file)
                    all_data.append(data) #all_data should be of length n and each element is a full time series of emotion data
        return all_data
    
    def resample_data(self, file_data): #resampling the data linearly so we have 10 readings per second
        #'100L' translates to 10 Hz frequency
        df = pd.DataFrame(file_data)
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        df = df.resample(self.target_freq).interpolate(method='linear')
        scores_array = np.stack(df['scores'].apply(lambda x: np.array(x)).values)

        return scores_array
    
    def prepare_and_segment_data(self):
        '''
        Method #1 for training model:
        Segments data so that only the previous 1 minute (600 readings) are used to make a guess for the next 1 minute
        data_array should be an np.matrix of size ~100xmax(logfilelength)x7 (100 = #time series')
        '''
        
        all_data = self.read_emotion_logs(self.log_dir)
        resampled_data = [self.resample_data(file_data, self.target_freq) for file_data in all_data]
        
        num_files = len(resampled_data)
        num_features = resampled_data[0].shape[1]  # Number of emotion scores -- should be 7
        max_length = max(file_data.shape[0] for file_data in resampled_data) #maximum time series length
        
        X, Y = [], []

        for file_data in resampled_data:
            for start in range(0, file_data.shape[0] - self.segment_length, self.stride):
                end = start + self.segment_length
                if end + self.segment_length <= file_data.shape[0]:
                    X.append(file_data[start:end])
                    Y.append(file_data[end:end + self.segment_length])
                    #Note some of X and Y are going to be mostly 0's rather than vectors
                    #X and Y should have size ~ # of minutes of data x600x7
        X = np.array(X)
        Y = np.array(Y)

        return X, Y
    
    

    def train_val_testing_split(self,X,Y):
        raise NotImplementedError
    



    #testing functions:
extractor = emotionFeatureExtractor()
XData, YData = extractor.prepare_and_segment_data()
