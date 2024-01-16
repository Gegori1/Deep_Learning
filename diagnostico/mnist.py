# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.svm import SVR
import joblib
from diagnostic_package.Utils import find_root_dir, read_config

# %% config global

rd = find_root_dir()

cf = read_config()

data_path = rd / cf['mnist']['data']['data_path']

train_path = data_path / cf['mnist']['data']['train_path_mnist']
test_path = data_path / cf['mnist']['data']['test_path_mnist']

scaler_filename = cf['mnist']['data']['scaler_filename']
model_filename = cf['mnist']['data']['model_filename']

# %% load data
data_train = pd.read_csv(train_path) 
data_test = pd.read_csv(test_path)

data_train.head()
data_test.head()

# %% check data pre

data_train.info() # check if nulls

data_train.describe() # check scales

# %%
X = data_train.iloc[:,1:]
y = data_train.iloc[:,0]

y.value_counts() # check if balanced

# %% split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# save scaler
# scaler_filename = "scaler.save"
joblib.dump(scaler, scaler_filename)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% train model
SVR = SVR(kernel = 'rbf', C = 1e3, gamma = 0.1)