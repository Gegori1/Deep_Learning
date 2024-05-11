
import os
import shutil

# ==============================================================================
# check unique endungen
def check_unique_endungen(dict_files):
    files = dict_files.keys()
    
    endungen = list({i.split(".")[-1] for i in files})
    
    return endungen

# ==============================================================================
# remove non-desired files from json
def remove_nonsupported(dict_files, extension):
    files = dict_files.keys()
    
    files = {i: dict_files[i] for i in files if i.split(".")[-1] != extension}
    
    return files

# ==============================================================================
# copy files
def copy_files(dict_files, org_folder, dst_folder, prefix):
    files = dict_files.keys()
    
    for i in files:
        file_name = prefix + "_" + i
        if not os.path.exists(os.path.join(dst_folder, file_name)):
            shutil.copy2(os.path.join(org_folder, i), os.path.join(dst_folder, file_name))
            
# ==============================================================================