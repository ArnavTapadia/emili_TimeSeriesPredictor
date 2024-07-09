import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split


class emotionFeatureExtractor:
    def __init__(self, log_dir = 'time_series_predictor/Data', segment_length=600, stride=600, target_freq='100ms'):
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
    
    def resample_data(self, file_data, resampling_method):
        print(resampling_method)
        '''
        resampling method is either
        1. exponential moving average with a half-life of say 500ms
        2. the sequence of pairs (timestamp_n, scores_n). No binning, no averaging
        '''


        #'100ms' translates to 10 Hz frequency
        df = pd.DataFrame(file_data) #current data is sampled at ~9-12 readings per second - varies
        df = df[['time', 'scores']] #only care about these 2
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True) #time is now the index
        nidx = pd.date_range(df.index.min(), df.index.max(), freq='100ms')
        uniondf = df.reindex(df.index.union(nidx))
        # uniondf.index = uniondf.index - uniondf.index[0]

        # Expand the 'scores' column into multiple columns
        scores_df = pd.DataFrame(df['scores'].tolist(), index=df.index) #Note features are normalized using sum
        # scores_df.index = scores_df.index - scores_df.index[0] #starting time at 0ms

        #TODO: Fix resampling

        # ––––––– Method 1: exponential moving average with a half-life of say 500ms –––––––––––––
        #resample taking mean of each bin and then apply ewma
        #adjust = False since data is irregularly spaced
        resampled_scores_df = scores_df.resample(self.target_freq).mean().ewm(halflife=500/1000, adjust=False).mean()
        
        resampled_scores_df2 = scores_df.resample(self.target_freq).apply(lambda x: x.ewm(halflife=500/1000).mean().iloc[-1] if not x.empty else None)
        
        
        

        testing = scores_df.loc[:,0]/1000000
        testingResampledlinear = testing.resample(self.target_freq).interpolate(method='linear')
        testingResamplednearest = testing.resample(self.target_freq).interpolate(method='nearest')
        # Resample and interpolate
        # resampled_scores_df = scores_df.resample(self.target_freq).mean().ffill()

        # ––––––– Method 3: add 0.1s times and interpolate then reindex ––––––––––––– 

        

        # ––––––– Method 2: predict the sequence of pairs (timestamp_n, scores_n). No binning, no averaging –––––––––


        # # Convert back to numpy array
        # scores_array = resampled_scores_df.to_numpy()
        scores_array = scores_df.to_numpy()
        scores_array = scores_array/np.sum(scores_array,axis=1)[:,np.newaxis] #normalizing to 0-1

        return scores_array
    
    def prepare_and_segment_data(self, resample_method = 'expmovavg'):
        '''
        Method #1 for training model:
        Segments data so that only the previous 1 minute (600 readings) are used to make a guess for the next 1 minute
        data_array should be an np.matrix of size ~100xmax(logfilelength)x7 (100 = #time series')
        '''
        
        all_data = self.read_emotion_logs()
        resampled_data = [self.resample_data(file_data, resample_method) for file_data in all_data]
        
        #for testing 3 lines below:
        # num_files = len(resampled_data)
        # num_features = resampled_data[0].shape[1]  # Number of emotion scores -- should be 7
        # max_length = max(file_data.shape[0] for file_data in resampled_data) #maximum time series length
        
        X, Y = [], []

        for file_data in resampled_data:
            for start in range(0, file_data.shape[0] - self.segment_length, self.stride): #increments of length 600(segment length), with the last 600 for the label
                end = start + self.segment_length
                if end + self.segment_length <= file_data.shape[0]:
                    X.append(file_data[start:end])
                    Y.append(file_data[end:end + self.segment_length])
                    #Note some of X and Y are going to be mostly 0's rather than vectors
                    #X and Y should have size ~ # of minutes of data x600x7
        X = np.array(X)
        Y = np.array(Y)
        #for each X[i], the corresponding predicted label is Y[i]
        #X.shape[0] ~ total number of minutes of data (slightly less)
        #Some data gets unused (if the time series was not a whole number of minutes long)
        #can be adjusted by changing stride to have overlapping segments
        return X, Y
    
    

    def train_val_testing_split(self,X,Y, split = [0.8,0.1,0.1], random_state = None):
        n = X.shape[0]
        #randomly select 80% for training, 10% for validation, 10% for testing
        assert sum(split) == 1 and split[0] > 0 and split [1] >= 0 and split[2] > 0

        # First split to create training and temp sets
        xTrain, xTemp, yTrain, yTemp = train_test_split(X, Y, test_size=split[1] + split[2], random_state= random_state)

        # Calculate validation and test split size proportionally from the temp set
        val_test_ratio = split[1] / (split[1] + split[2])

        #split again
        xVal, xTest, yVal, yTest = train_test_split(xTemp, yTemp, test_size=1-val_test_ratio + split[2], random_state= random_state)


        return xTrain, yTrain, xVal, yVal, xTest, yTest
    
    def save_sampleData(self, x_train, y_train, x_val, y_val, x_test, y_test, save_dir = 'time_series_predictor/Data/Data_Saves' , fName = ''):
        # Save each array to a file
        np.save(os.path.join(save_dir, fName + 'x_train.npy'), x_train)
        np.save(os.path.join(save_dir, fName + 'y_train.npy'), y_train)
        np.save(os.path.join(save_dir, fName + 'x_val.npy'), x_val)
        np.save(os.path.join(save_dir, fName + 'y_val.npy'), y_val)
        np.save(os.path.join(save_dir, fName + 'x_test.npy'), x_test)
        np.save(os.path.join(save_dir, fName + 'y_test.npy'), y_test)


    #TODO: write feature extraction for padding and masking method

extractor = emotionFeatureExtractor()
XData,YData = extractor.prepare_and_segment_data(resample_method='expmovavg')
xTr,yTr,xV,yV,xTest,yTest = extractor.train_val_testing_split(XData,YData, random_state=5)