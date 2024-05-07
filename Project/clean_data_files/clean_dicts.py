# %%
import json
import os

# %% parameters
folder_indicators = "Indicators"
group = "female"
new_indicator = f"coordinates_{group}_global.json"



def open_indicator(name_indicator):
    indicator_path = os.path.join(folder_indicators, name_indicator)
    with open(indicator_path, "r") as file:
        dict_files = json.load(file)
    return dict_files

def remove_nonsupported(dict_files, extension):
    dict_files = {k: v for k, v in dict_files.items() if not k.endswith(extension)}
    return dict_files

def give_prefix(dict_files, prefix):
    dict_files = {prefix + "_" + k: v for k, v in dict_files.items()}
    return dict_files

# %% import indicator function
dict_file_names = os.listdir(folder_indicators)
dict_file_names = [name for name in dict_file_names if "women" in name]
# dict_file_names = [name for name in dict_file_names if "men" in name]
# dict_file_names = [name for name in dict_file_names if "women" not in name]

new_dict = {}
for name in dict_file_names:
    dict_files = open_indicator(name)
    dict_files = remove_nonsupported(dict_files, "gif")
    prefix = name.split("__")[0]
    dict_files = give_prefix(dict_files, prefix)
    new_dict.update(dict_files)
    
# %% save new indicator
new_indicator_path = os.path.join(folder_indicators, new_indicator)
with open(new_indicator_path, "w") as file:
    json.dump(new_dict, file)

# %%
