#%%
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class emotionFeatureExtractor:
    def __init__(self, log_dir = '../Data/Data_Saves', segment_length=600, stride=600, target_freq='100ms'):
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

        # Expand the 'scores' column into multiple columns
        scores_df = pd.DataFrame(df['scores'].tolist(), index=df.index) #Note features are normalized using sum

        #TODO: Find best resampling
        if resampling_method == 'ewma':
            # ––––––– Method 1: resample, mean then exponential moving average with a half-life of say 500ms –––––––––––––
            #resample taking mean of each bin and then apply ewma
            #adjust = False since data is irregularly spaced
            resampled_scores_df = scores_df.resample(self.target_freq).mean().ewm(halflife=500/1000, adjust=False).mean()
       
        elif resampling_method == 'binnedewma':
            # ––––––– Method 2: resample with exponential moving average with a half-life of say 500ms for every bin–––––––––––––
            #https://stackoverflow.com/questions/66271048/dataframe-ewm-for-groupby-resample
            #apply ewm to each 100ms bin (i think?)
            #realistically just takes nearest value -> probs not useful & super slow -- shouldn't use
            #TODO: check if this is actually good -- can replace with interpolation method 3 -> ewm
            resampled_scores_df = scores_df.resample(self.target_freq).apply(lambda x: x.ewm(halflife=500/1000, adjust=False).mean().iloc[-1] if not x.empty else None)
        
        elif resampling_method == 'interpolation':
            # ––––––– Method 3: add 0.1s times and interpolate then reindex ––––––––––––– 
            nidx = pd.date_range(df.index.min(), df.index.max(), freq='100ms')
            uniondf = scores_df.reindex(scores_df.index.union(nidx)) #adding every 100ms as times with nans
            resampled_scores_df = uniondf.interpolate(method='time') #using linear interpolation
            #cast out non multiples of 100ms
            resampled_scores_df = resampled_scores_df.loc[nidx]

        elif resampling_method == 'ewmainterp':
            # ––––––– Method 4: add 0.1s times and use ewma to smooth ––––––––––––– 
            nidx = pd.date_range(df.index.min(), df.index.max(), freq='100ms')
            uniondf = scores_df.reindex(scores_df.index.union(nidx)) #adding every 100ms as times with nans
            resampled_scores_df = uniondf.ewm(halflife=500/1000, adjust=False, ignore_na=True).mean()
            #cast out non multiples of 100ms
            resampled_scores_df = resampled_scores_df.loc[nidx]
        
        elif resampling_method == 'interp_ewmaSmooth':
            # ––––––– Method 5: add 0.1s times and interpolate then apply ewma smoothing ––––––––––––– 
            nidx = pd.date_range(df.index.min(), df.index.max(), freq='100ms')
            uniondf = scores_df.reindex(scores_df.index.union(nidx)) #adding every 100ms as times with nans
            resampled_scores_df = uniondf.interpolate(method='time') #using linear interpolation
            resampled_scores_df = resampled_scores_df.ewm(halflife=500/1000, adjust=False).mean()
            #cast out non multiples of 100ms
            resampled_scores_df = resampled_scores_df.loc[nidx]

        elif resampling_method == 'times_scores':
            # ––––––– Method 6: predict the sequence of pairs (timestamp_n, scores_n). No binning, no averaging –––––––––
            resampled_scores_df = scores_df.reset_index()
            resampled_scores_df['time'] = resampled_scores_df['time']-resampled_scores_df['time'][0]
            resampled_scores_df['time'] = resampled_scores_df['time'].dt.total_seconds()

        # Convert back to numpy array
        #reset scores time to 0
        scores_array = resampled_scores_df.to_numpy()
        #TODO: determine if this is the best way to normalize or I should divide by 1000000
        if resampling_method != 'times_scores':
            scores_array = scores_array/(np.sum(scores_array,axis=1)[:,np.newaxis]) #normalizing to 0-1
        else:
            times = scores_array[:,0]
            scores_array = scores_array[:,1:]/(np.sum(scores_array[:,1:],axis=1)[:,np.newaxis]) #normalizing to 0-1
            scores_array = np.hstack((times[:,np.newaxis],scores_array))
            

        return scores_array
    
    def prepare_and_segment_data(self, resample_method = 'ewma'):
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
            #TODO: determine best way to split data when using timestamps and scores as input and change model
            if resample_method != 'times_scores':
                for start in range(0, file_data.shape[0] - self.segment_length, self.stride): #increments of length 600(segment length), with the last 600 for the label
                    end = start + self.segment_length
                    if end + self.segment_length <= file_data.shape[0]:
                        X.append(file_data[start:end])
                        Y.append(file_data[end:end + self.segment_length])
                        #Note some of X and Y are going to be mostly 0's rather than vectors
                        #X and Y should have size ~ # of minutes of data x600x7
            else:
                for start in range(0,int(np.ceil(file_data[-1,0])),self.stride//10):
                    end = start + self.stride/10 #first 60 s of data
                    if np.shape(file_data[(file_data[:,0] >= end) & (file_data[:,0]<=end+self.stride/10)])[0]:
                        
                        #ensuring the Y layer will have at least 1 minute of data

                        X.append(file_data[(file_data[:,0] >= start) & (file_data[:,0]<=end)])
                        Y.append(file_data[(file_data[:,0] >= end) & (file_data[:,0]<=end+self.stride/10)])
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
    
    def save_trainTestData(self, x_train, y_train, x_val, y_val, x_test, y_test, save_dir = '../Data/Data_Saves/Preprocessed', fName = ''):
        # Save each array to a file
        np.save(os.path.join(save_dir, fName + '_x_train.npy'), x_train)
        np.save(os.path.join(save_dir, fName + '_y_train.npy'), y_train)
        np.save(os.path.join(save_dir, fName + '_x_val.npy'), x_val)
        np.save(os.path.join(save_dir, fName + '_y_val.npy'), y_val)
        np.save(os.path.join(save_dir, fName + '_x_test.npy'), x_test)
        np.save(os.path.join(save_dir, fName + '_y_test.npy'), y_test)
    
    def save_PreprocessedData(self, X, Y, save_dir = '../Data/Data_Saves/Preprocessed' , fName = ''):
        # Save each array to a file
        np.save(os.path.join(save_dir, fName + '_x.npy'), X)
        np.save(os.path.join(save_dir, fName + '_y.npy'), Y)

    def update_dataSaves(self):
        ''' 
        Function to read the json data files and update the saved numpys for XData and YData
        To be used mainly for saving data to test preprocessing methods
        '''
        for filterMethod in ['ewma', 'binnedewma', 'interpolation', 'ewmainterp', 'interp_ewmaSmooth', 'times_scores']:
            XData,YData = self.prepare_and_segment_data(resample_method=filterMethod)
            self.save_PreprocessedData(X=XData,Y=YData, fName=filterMethod, save_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'Data/Data_Saves/Preprocessed'))

    def compareFilterMethods(self, filterMethodToComp = ['ewma', 'binnedewma', 'interpolation', 'ewmainterp', 'interp_ewmaSmooth'], iFile = -1):
        all_data = self.read_emotion_logs()
        times_scores_data = [self.resample_data(file_data, 'times_scores') for file_data in all_data]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(filterMethodToComp)+1))

        if iFile not in range(len(all_data) - 1):
            iFile = np.random.randint(0, len(all_data) - 1) #file we want to compare
        actualData = times_scores_data[iFile]  # correct data
        actualData_time = actualData[:, 0]
        actualData_features = actualData[:, 1:]  # Exclude time column
        num_features = actualData_features.shape[1]

        fig, axes = plt.subplots(num_features, 1, figsize=(10, 3 * num_features), sharex=True)
        fig.suptitle(f'Features vs Time for Resample Method, file {iFile}')

        #plot actual data
        for i in range(num_features):
            axes[i].plot(actualData_time, actualData_features[:, i], label='Actual Data', marker = 'o', linestyle = '-', markersize=3, color=colors[0])

        colors = colors[1:]
        for resample_method, color in zip(filterMethodToComp, colors):
            resampled_data = [self.resample_data(file_data, resample_method) for file_data in all_data]  # filter by chosen method

            # Make plots of actual_data and resampled_data
            resample_data_features = resampled_data[iFile]  # resampled data
            resampled_data_time = np.arange(0, 0.1 * np.shape(resample_data_features)[0], 0.1)  # adding time increments to data without time
            resampled_data_time = resampled_data_time[:np.shape(resample_data_features)[0]]
            # Number of features
            num_features = actualData_features.shape[1]

            # Plot each feature for each resample method
            for i in range(num_features):
                axes[i].plot(resampled_data_time, resample_data_features[:, i], label=resample_method, color=color, linestyle='--', marker = 'o', markersize=2)
                
        #label subplots
        for i in range(num_features):       
            axes[i].set_ylabel(f'Feature {i + 1}')
            axes[i].legend()
            axes[i].grid(True)

    
        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Adjust title position
        axes[-1].set_xlim(0, 100)
        plt.show()


        
    
    #TODO: write feature extraction for padding and masking method

# extractor = emotionFeatureExtractor(log_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'Data/Data_Saves'))
# #%% Comparing filter methods
# %matplotlib widget
# # extractor.compareFilterMethods(['ewma', 'ewmainterp'], iFile = 65)
# # extractor.compareFilterMethods(['interpolation', 'interp_ewmaSmooth'], iFile = 65)
# # extractor.compareFilterMethods(['ewma', 'ewmainterp', 'interpolation'], iFile = 10)
# # extractor.compareFilterMethods()
# extractor.compareFilterMethods(['ewma', 'ewmainterp', 'interpolation'], iFile = 68)


# extractor.update_dataSaves()
# for meth in ['ewma', 'binnedewma', 'interpolation', 'ewmainterp', 'interp_ewmaSmooth', 'times_scores']:
#     XData,YData = extractor.prepare_and_segment_data(resample_method=meth)

# xTr,yTr,xV,yV,xTest,yTest = extractor.train_val_testing_split(XData,YData, random_state=5)
# %%
