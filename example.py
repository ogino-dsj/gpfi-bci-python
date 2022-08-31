# -*- coding: utf-8 -*-
"""
Obtaining Feature Importance by using Group Permutation Feature Importance (GPFI)
for BCI applications

@author: Mikito Ogino
"""

import GPFI
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

X = pd.read_csv("X.csv", header=None) #(1000 epochs, 63 channels x 50 time samples)
Y = pd.read_csv("Y.csv", header=None) #(1000 epochs, )
channels = pd.read_csv("channels.csv", header=None)

X_train, X_test, Y_train, Y_test = train_test_split(X.values, np.reshape(Y.values,(-1)) ,train_size=0.8)

channels = np.reshape(channels.values,(-1))

clf = LinearDiscriminantAnalysis()
clf.fit(X_train, Y_train)

gpfi = GPFI.GPFI()
gpfi.fit(X_test, Y_test, model = clf, channels = channels, temporal_group = 5)

print(gpfi.channelFI)
print(gpfi.temporalFI)

