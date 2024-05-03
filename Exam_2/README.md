# Actions identification using Recursive Neural Networks

This repository contains the code for the creation of a neural network capable of identifying multiple actions perform in a series of videos, portraiting 5 actions, one per video.

All the files can be obtained through the following [google drive](https://drive.google.com/drive/folders/1Alm0gl_KaDWFaN1m7nMa0y6q6rGqTR8d?usp=sharing) folder

The project has the following structure:

```
EXAM_2
├── dataloader.py
├── dataset_loader.py
├── fit_func.py
├── README.md
├── TestActivityIdentif.ipynb
├── TrainActivityIdentif.ipynb
└── video_sequence_model.py
```

Where the notebooks contain all the training and testing logic and the rest of the files are called inside these files to run the pipeline.


## Description

For the data loading process 10 frames were cropped randomly from each video. These were later transformed with the help of pytorch `transforms` library to augment the data and avoid overfitting.

The data was dosed to a model with 10 `MobileNet` Neural Networks. 

For the training process two approximations were used. The first one used the first 5 frames to input to the LSTM nural network. The second one used all 10 frames to feed the LSTM neural network. The best model was based on the validation multiclass accuracy.

## Results

Due to a small dataset, a test set was extracted from the train and validation dataset.
Each model was tested with 4 randomly selected subsets. The results are shown down below:

Aproximation 1 (first 5):

| accuracy | recall |
|----------|--------|
| 0.8      | 0.8    |
| 1        | 0.7    |
| 1        | 0.9    |
| 1        | 0.8    |

Approximation 2 (all 10):

| accuracy | recall |
|----------|--------|
| 1        | 0.8    |
| 1        | 0.9    |
| 1        | 0.7    |
| 1        | 0.9    |

## Discusion

As expected, the accuracy and recall are high, since the sample is taken from the train and validation dataset.

For later iterations it would be intersting to try with larger datasets.