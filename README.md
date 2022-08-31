# gpfi-bci-python
 Python library for calculating gpf which can ivisualize feature importance for nonlinear machine learning models

# Dependencies

- Python (>= 3.8)
- numpy (>= 1.22.3)
- pandas (>=1.0.5)
- scikit-learn (>= 1.0.2)

# Simple Demo (example.py)
```
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
```
