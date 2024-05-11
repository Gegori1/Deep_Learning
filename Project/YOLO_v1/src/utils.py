# %%
import os
import json
import pandas as pd

# %%

def indicator_to_df(current_folder_path, female_file_name, male_file_name)->pd.DataFrame:
    
    with open(os.path.join(current_folder_path, female_file_name)) as f:
        female_data = json.load(f)
    female_data = [(key, value[0], value[1], value[2], value[3],) for key, value in female_data.items()]
    female_df = pd.DataFrame(female_data, columns=["file_name", "x_min", "y_min", "x_max", "y_max"])
    female_df['gender'] = 'Female faces'

    with open(os.path.join(current_folder_path, male_file_name)) as f:
        male_data = json.load(f)
    male_data = [(key, value[0], value[1], value[2], value[3]) for key, value in male_data.items()]
    male_df = pd.DataFrame(male_data, columns=["file_name", "x_min", "y_min", "x_max", "y_max"])
    male_df['gender'] = 'Male faces'

    train_eval_df = pd.concat([female_df, male_df])
    
    not_files = [
        "f2_00001929.jpg", "f1_tr_75.jpg", "f1_tr_32.jpg", 
        "f1_tr_21.jpg", "f2_00001337.jpeg", "f2_00001831.jpg",
        "f2_missclassed (59).jpg", "f2_00001845.jpg", "f2_00000076.jpg",
        "f2_00000677.jpg", "f2_00001862.jpg", "f3_1 (770).jpg", "f2_00002183.jpg",
        "f2_00002200.jpg"
    ]
    
    train_eval_df = train_eval_df.query("file_name not in @not_files")
    
    return train_eval_df

# %%
if __name__ == "__main__":
    current_folder_path = os.path.join(os.getcwd(), "..", "Indicators/")
    female_file_name = "coordinates_female_global.json"
    male_file_name = "coordinates_male_global.json"
    
    ha = indicator_to_df(
        current_folder_path, 
        female_file_name,
        male_file_name
    )

# %%
