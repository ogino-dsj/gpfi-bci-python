# -*- coding: utf-8 -*-
"""
Obtaining Feature Importance by using Group Permutation Feature Importance (GPFI)
for BCI applications

@author: Mikito Ogino
"""

import numpy as np
import pandas as pd
import copy
from sklearn.metrics import accuracy_score

class GPFI:
    channelFI = {}
    temporalFI = {}
    
    def FeatureImportance(self, X_test, Y_test, model, channels, group):
        dimension = int(X_test.shape[1]/len(channels))
        columns = []
        for channel in channels:
            for j in range(1,dimension+1):
                columns.append(channel+"_"+str(j))
                
        Y_pred = model.predict(X_test)
        baseLine = accuracy_score(Y_test, Y_pred)
        
        diffs = []
        for single_group in group:
            pd_X = copy.deepcopy(pd.DataFrame(X_test, columns=columns))
            for instance in single_group:
                if instance in columns:
                    pd_X[instance] = pd_X[instance].sample(frac = 1).reset_index(drop = True)
            Y_pred = model.predict(pd_X.values)
            acc = accuracy_score(Y_test, Y_pred)
            diffs.append(-(acc-baseLine))
        
        return diffs
    
    def fit(self, X_test, Y_test, model, channels, temporal_group = 1):
        """
        Calculating feature importance for EEG channel groups and temporal groups
        
        Parameters
        ----------
        X_test : ndarray (epochs, channel x time)
            Collection of feature vectors
        Y_test : ndarray (epochs,)
            Collection of labels for the feature vectors
        model : objecct of sklearn
        channels : list
            List of names of EEG channels
        temporal_group : int
            Number of temporal features to be grouped
        """
        
        if(X_test.shape[1] % len(channels) !=0):
            raise ValueError(
                "The dimensions of X_test should be multiple of the number of channels"
            )
        
        if(X_test.shape[1] % temporal_group !=0):
            raise ValueError(
                "The dimensions of X_test should be multiple of the temporal_dimension"
            )
            
        dimension = int(X_test.shape[1]/len(channels))

        #Make the group for EEG channels
        groups = []
        for channel in channels:
            group = []
            for j in range(1,dimension+1):
                group.append(channel+"_"+str(j))
            groups.append(group)
        diffs = self.FeatureImportance(X_test, Y_test, model, channels, groups)
        for ch, diff in zip(channels, diffs):
            self.channelFI[ch] = diff
    
        #Make the group for temporal channels
        groups = []
        for i in range(1,dimension+1,temporal_group):
            group = []
            for j in range(i,i+5):
                for channel in channels:
                    group.append(channel+"_"+str(j))
            groups.append(group)
        diffs = self.FeatureImportance(X_test, Y_test, model, channels, groups)
    
        for index, diff in enumerate(diffs):
            self.temporalFI[index] = diff
            
            
        