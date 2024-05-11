# %%
import os
import json

# %%

class UpdateIndicators:
    def __init__(self, src_file: str, dst_file: str, folder_match: str, path_to_data: str)->None:
        self.src_file = src_file
        self.dst_file = dst_file
        self.folder_match = folder_match
        self.path_to_data = path_to_data
        self.folder_path = os.path.join(path_to_data, folder_match)
        self.src_path = os.path.join(path_to_data, src_file)
        self.dst_path = os.path.join(path_to_data, dst_file)
        
    def copy_indicators(self)->None:
        file_names = os.listdir(self.folder_path)
        with open(self.src_path, 'r') as f:
            indicators = json.load(f)
        to_move = {}
        for key, value in indicators.items():
            if key in (file_names):
                to_move[key] = value
        for key in to_move.keys():
            indicators.pop(key)
        with open(self.dst_path, 'w') as f:
            json.dump(to_move, f)
            
        self.to_move = to_move
        self.indicators = indicators
            
    def rename_original(self, new_name: str)->None:
        save_path = os.path.join(self.path_to_data, new_name)
        with open(save_path, 'w') as f:
            json.dump(self.indicators, f)
        
    def remove_original(self)->None:
        os.remove(self.src_path)
    
