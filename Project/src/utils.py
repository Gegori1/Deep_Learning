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
