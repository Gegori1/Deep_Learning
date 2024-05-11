# %%
import os
import pandas as pd
import cv2
from time import sleep
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# %% Load the data
data = pd.read_csv('face_data.csv')
# not_files = [
#     "f2_00001929.jpg", "f1_tr_75.jpg", "f1_tr_32.jpg", 
#     "f1_tr_21.jpg", "f2_00001337.jpeg", "f2_00001831.jpg",
#     "f2_missclassed (59).jpg", "f2_00001845.jpg", "f2_00000076.jpg",
#     "f2_00000677.jpg", "f2_00001862.jpg", "f3_1 (770).jpg"
# ]
not_files = ["f2_00002200.jpg"]

data.query("file_name in @not_files", inplace=True)

# datan = data.query("x_max - x_min < 0.3 and y_max - y_min < 0.3")
# %%

for i, row in data.iterrows():
    fig, ax = plt.subplots()
    img_path = os.path.join(
            "data", "Archive_joint", row['gender'], row['file_name']
    )
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    ax.imshow(img)
    
    ax.add_patch(
        Rectangle(
            (row['x_min']*w, row['y_min']*h),
            (row['x_max'] - row['x_min'])*w,
            (row['y_max'] - row['y_min'])*h,
            edgecolor='r',
            facecolor='none'
        )
    )
    print(f"gender: {row['gender']}")
    print(f"path: {row['file_name']}")
    plt.show()
    sleep(1)
# %%
