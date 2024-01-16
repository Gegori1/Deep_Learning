# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score
)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier

from diagnostic_package.Utils import find_root_dir, read_config
# %% config global

rd = find_root_dir() # finds root dir
cf = read_config() # read configuration

# data and models dictionary
ddict, mdict = cf['mnist']['data'], cf['mnist']['models']

# data path
data_path = rd / ddict['data_path']
train_path = data_path / ddict['data_train']
test_path = data_path / ddict['data_test']

# model path
model_path = rd / mdict['model_path']
model_filename = model_path / mdict['model_filename']

# %% load data
data_train = pd.read_csv(train_path, header=None) 
data_test = pd.read_csv(test_path, header=None)

data_train.head()
data_test.head()

# %% check data pre

data_train.info() # check if nulls

# check scales
data_train.describe().T["min"].unique()
data_train.describe().T["max"].unique()

# %%
X = data_train.iloc[:,1:]
y = data_train.iloc[:,0]

# check if balanced
classes = y.value_counts()
plt.bar(classes.index, classes.values)
plt.show()

# %% split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# %% max scaler
X_train = X_train/255
X_test = X_test/255

# %% select features
for l in range(1, y.nunique() - 1):
    lda = LinearDiscriminantAnalysis(n_components=l)
    lda.fit(X_train, y_train)
    print("Components: ", l, " - Score: ", lda.score(X_test, y_test))
    
lda = LinearDiscriminantAnalysis(n_components=1).fit(X_train, y_train)
  
# %% get feature importance of tree model
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree.feature_importances_
plt.barh(X_train.columns, tree.feature_importances_)

# get columns with importance > 0.01
cols = X_train.columns[tree.feature_importances_ > 0.01]
cols.shape
# %% train and test model_svc
X_train = X_train[cols].assign(lda_ = lda.transform(X_train))
X_test = X_test[cols].assign(lda_ = lda.transform(X_test))

# %% train and test model
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred_test_proba = model.predict_proba(X_test)
y_pred_train_proba = model.predict_proba(X_train)

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)


# visualize accuracy and confusion matrix
print(accuracy_score(y_test, y_pred_test))
print(confusion_matrix(y_test, y_pred_test))

roc_auc_score(y_test, y_pred_test_proba, multi_class='ovr', average='macro')
roc_auc_score(y_train, y_pred_train_proba, multi_class='ovr', average='macro')

# The model is overfitting. Fine tuning is needed.
# %% save model

# lda
lda_filename = model_path / "lda.joblib"
joblib.dump(lda, lda_filename)

# col
cols_filename = model_path / "cols.joblib"
joblib.dump(cols, cols_filename)

joblib.dump(model, model_filename)
