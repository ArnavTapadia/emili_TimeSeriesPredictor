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
    
    

    def non_overlappingSegmentData(data_array, segmentLength = 600,stride = 600):
        raise NotImplementedError
