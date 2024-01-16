import joblib
from diagnostic_package.Utils import find_root_dir, read_config

# %% config global

rd = find_root_dir() # finds root dir
cf = read_config() # read configuration

# data and models dictionary
ddict, mdict = cf['mnist']['data'], cf['mnist']['models']


# %%

# data path
data_path = rd / ddict['data_path']
test_path = data_path / ddict['data_test']

# model path
model_path = rd / mdict['model_path']
scaler = model_path / mdict['scale']
model_filename = model_path / mdict['model_filename']