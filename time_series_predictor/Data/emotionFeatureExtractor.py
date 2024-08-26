#%%
import numpy as np
import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random


class emotionFeatureExtractor:
    def __init__(self, log_dir = '../Data/Data_Saves', y_prediction_length = 600, x_segment_length=600, stride=600, target_freq='100ms'):
        self.log_dir = log_dir
        self.x_segment_length = x_segment_length
        self.stride = stride
        self.target_freq = target_freq
        self.y_prediction_length = y_prediction_length


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

    def segment_data(self, data, resample_method):
        X, Y = [], []
        for file_data in data:
            if resample_method != 'times_scores':
                for start in range(0, file_data.shape[0] - self.x_segment_length, self.stride):
                    end = start + self.x_segment_length
                    if end + self.y_prediction_length <= file_data.shape[0]:
                        X.append(file_data[start:end])
                        Y.append(file_data[end:end + self.y_prediction_length])
            else:
                for start in range(0, int(np.ceil(file_data[-1,0])), self.stride//10):
                    end = start + self.x_segment_length/10
                    if np.shape(file_data[(file_data[:,0] >= end) & (file_data[:,0]<=end+self.y_prediction_length/10)])[0] > self.y_prediction_length: #TODO: Fix this
                        X.append(file_data[(file_data[:,0] >= start) & (file_data[:,0]<=end)])
                        Y.append(file_data[(file_data[:,0] >= end) & (file_data[:,0]<=end+self.y_prediction_length/10)])
        return np.array(X), np.array(Y) 
        
    def train_val_testing_split(self, data, split=[0.8, 0.1, 0.1], random_state=None):
        n = len(data)
        assert sum(split) == 1 and split[0] > 0 and split[1] >= 0 and split[2] > 0
        #permute the data according to random_state
        random.Random(random_state).shuffle(data)
        #calculate length of each time series in terms of segment length
        nSegments = np.array([(sample.shape[0]-self.y_prediction_length)//self.x_segment_length for sample in data])
        cumNSegments = np.cumsum(nSegments) 
        
        # Splitting according to split variable
        # take first split[0] for train, split[1] for val, split[2] for test
        trainSegmentCount = cumNSegments[-1]*split[0]
        valSegmentCount = cumNSegments[-1]*split[1]

        #masks for train val test split
        bTrain = cumNSegments < trainSegmentCount
        bVal = (trainSegmentCount < cumNSegments) & (cumNSegments < trainSegmentCount + valSegmentCount)
        bTest = ~(bTrain | bVal)

        train_data = [data[i] for i in range(len(bTrain)) if bTrain[i]]
        val_data = [data[i] for i in range(len(bVal)) if bVal[i]]
        test_data = [data[i] for i in range(len(bTest)) if bTest[i]]

        return train_data, val_data, test_data
    
    def load_from_data_saves(self, resample_method='ewma'):
        '''
        method to load data from pre-saved folder
        '''
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data/Data_Saves/Preprocessed')

        # Load the segmented data
        X_train = np.load(os.path.join(save_dir, f'{resample_method}_X_train.npy'), allow_pickle=True)
        Y_train = np.load(os.path.join(save_dir, f'{resample_method}_Y_train.npy'), allow_pickle=True)
        X_val = np.load(os.path.join(save_dir, f'{resample_method}_X_val.npy'), allow_pickle=True)
        Y_val = np.load(os.path.join(save_dir, f'{resample_method}_Y_val.npy'), allow_pickle=True)
        X_test = np.load(os.path.join(save_dir, f'{resample_method}_X_test.npy'), allow_pickle=True)
        Y_test = np.load(os.path.join(save_dir, f'{resample_method}_Y_test.npy'), allow_pickle=True)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test



    def update_dataSaves(self, split = [0.8, 0.1, 0.1], random_state=np.random.randint(0,2**30)):
        all_data = self.read_emotion_logs()
        #obtains a train val and testing split
        #should be such that all the data is saved in the same split random state

        for filterMethod in ['ewma', 'interpolation', 'ewmainterp', 'interp_ewmaSmooth', 'times_scores']:
            #data imputation
            resampled_data = [self.resample_data(file_data, filterMethod) for file_data in all_data]
            resampled_train_data, resampled_val_data, resampled_test_data = self.train_val_testing_split(data = resampled_data, random_state=random_state, split=split)

            #then segments the data correctly according to the stride required
            X_train, Y_train = self.segment_data(resampled_train_data, filterMethod)
            X_val, Y_val = self.segment_data(resampled_val_data, filterMethod)
            X_test, Y_test = self.segment_data(resampled_test_data, filterMethod)

            save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data/Data_Saves/Preprocessed')
            # Save the segmented data
            np.save(os.path.join(save_dir, f'{filterMethod}_X_train.npy'), X_train)
            np.save(os.path.join(save_dir, f'{filterMethod}_Y_train.npy'), Y_train)
            np.save(os.path.join(save_dir, f'{filterMethod}_X_val.npy'), X_val)
            np.save(os.path.join(save_dir, f'{filterMethod}_Y_val.npy'), Y_val)
            np.save(os.path.join(save_dir, f'{filterMethod}_X_test.npy'), X_test)
            np.save(os.path.join(save_dir, f'{filterMethod}_Y_test.npy'), Y_test)
    
    def get_data_split(self, resample_method = 'ewmainterp', split = [0.8, 0.1, 0.1], random_state=np.random.randint(0,2**30)):
        '''
        Similar to updata_dataSaves
        Just returns to user the requested data while processing it in real time - slower
        '''

        all_data = self.read_emotion_logs()
        #obtains a train val and testing split
        #should be such that all the data is saved in the same split random state

    
        #data imputation
        resampled_data = [self.resample_data(file_data, resample_method) for file_data in all_data]
        resampled_train_data, resampled_val_data, resampled_test_data = self.train_val_testing_split(data=resampled_data, random_state=random_state, split=split)

        #then segments the data correctly according to the stride required
        X_train, Y_train = self.segment_data(resampled_train_data, resample_method)
        X_val, Y_val = self.segment_data(resampled_val_data, resample_method)
        X_test, Y_test = self.segment_data(resampled_test_data, resample_method)

        return X_train, Y_train, X_val, Y_val, X_test, Y_test



    def compareFilterMethods(self, filterMethodToComp = ['ewma', 'binnedewma', 'interpolation', 'ewmainterp', 'interp_ewmaSmooth'], iFile = -1):
        #Function to be used to view the results of different data imputation methods
        
        all_data = self.read_emotion_logs()
        times_scores_data = [self.resample_data(file_data, 'times_scores') for file_data in all_data]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(filterMethodToComp)+1))

        if iFile not in range(len(all_data) - 1):
            iFile = np.random.randint(0, len(all_data) - 1) #file we want to compare
        actualData = times_scores_data[iFile]  # correct data
        actualData_time = actualData[:, 0]
        actualData_features = actualData[:, 1:]  # Exclude time column
        num_features = actualData_features.shape[1]

        fig, axes = plt.subplots(num_features, 1, figsize=(10, 3 * num_features), sharex=True, sharey = True)
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
        axes[-1].set_ylim(0, 1)
        plt.show()


extractor = emotionFeatureExtractor(log_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'Data/Data_Saves'))
#%% Comparing filter methods
# %matplotlib widget
# # extractor.compareFilterMethods(['ewma', 'ewmainterp'], iFile = 65)
# # extractor.compareFilterMethods(['interpolation', 'interp_ewmaSmooth'], iFile = 65)
# # extractor.compareFilterMethods(['ewma', 'ewmainterp', 'interpolation'], iFile = 10)
# extractor.compareFilterMethods()
# extractor.compareFilterMethods(['ewma', 'interpolation', 'interp_ewmaSmooth','ewmainterp'], iFile = 68)


# extractor.update_dataSaves()
xTr,yTr,xVal,yVal,xTest,yTest = extractor.get_data_split(resample_method='ewmainterp', random_state=5)
xTr2,yTr2,xVal2,yVal2,xTest2,yTest2 = extractor.get_data_split(resample_method='ewma', random_state=5)


fig, axes = plt.subplots(1, 1, figsize=(15, 7), sharex=True, sharey=True)
fig.suptitle('KL Divergence vs Time Averaged Across Samples in Validation Set')


# %%
iSample = np.random.randint(0,xTest.shape[0])
num_features = 7
fig, axes = plt.subplots(num_features, 1, figsize=(10, 20), sharex=True, sharey=True)
fig.suptitle(f'xTest Features vs Time per Resample Method, iSample {iSample}')
colors = plt.cm.rainbow(np.linspace(0, 1, 2))

resampled_data_time = np.arange(0, 0.1 * np.shape(xTest)[1], 0.1)
resampled_data_time = resampled_data_time[:np.shape(xTest)[1]]

for i in range(num_features):
    axes[i].plot(resampled_data_time, xTest[iSample, :, i], label='ewmainterp', color=colors[0], linestyle='--', marker = 'o', markersize=2)
    axes[i].plot(resampled_data_time, xTest2[iSample, :, i], label='ewma', color=colors[1], linestyle='--', marker = 'o', markersize=2)


#label subplots
for i in range(num_features):       
    axes[i].set_ylabel(f'Feature {i + 1}')
    axes[i].legend()
    axes[i].grid(True)


axes[-1].set_xlabel('Time')
plt.tight_layout()
plt.subplots_adjust(top=0.95)  # Adjust title position
axes[-1].set_xlim(0, 60)
axes[-1].set_ylim(0,1)
plt.show()
# %%
