# %%
import json
import os

from copy_files_utils import(
    check_unique_endungen,
    remove_nonsupported,
    copy_files
)

# %% parameters
folder_indicators = "Indicators"
indicator_file = "coordinates_women_archive_2.json" #
new_folder = "Archive_joint"
origin_folder = os.path.join("Archive_2", "Female Faces") #
destination_folder = os.path.join("Archive_joint", "Female faces") #
prefix = "f3"

# %% import indicator function
indicator_path = os.path.join(folder_indicators, indicator_file)
with open(indicator_path, "r") as file:
    dict_files = json.load(file)
    
# %% check unique extensions
endungen = check_unique_endungen(dict_files)
print(endungen)

# %% remove non-desired files
extension = "gif"
dict_files = remove_nonsupported(dict_files, extension)

# %% copy files
copy_files(dict_files, origin_folder, destination_folder, prefix)
# %%

