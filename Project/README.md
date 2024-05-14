# Object detection and Verification project

All the notebooks, models and files are stored in the following [Google Drive](https://drive.google.com/drive/folders/1cd-1iBIo9-hp2N2NgeRFiL3ruwB9j1Lq?usp=sharing) repository.


The following repository has the following directory structure:

```
├───FasterRCNN-ResnetInceptionv1
│   └───src
├───Indicators
├───Presentation
├───Report
└───YOLO_v1
    ├───clean_data_files
    └───src
```

## Object detection

- YOLO v1

The initial intent was to use YOLO v1 for the face detection algorithm, but due to poor quality inference, the descicion to switch to a pretrained model was made.

- Faster R-CNN

A pretrained faster R-CNN model was fine tuned with the Female and Male dataset during two epochs obtaining an Average Precision (AP) of `0.5385` and Average Recall (AR) of `0.5805` on the validation set.

A first visual check indicates that the model worked well on the test set. An AP of `0.7245` and a AR of `0.5855` was obtained for this dataset.

## Object recognition

- FaceNet (ResnetInception v1)

A pretrained ResnetInception model was used to get a 512 dimensional embedding of each image. The distance between the resulting vectors was measured using the cosine distance.


## References

[FaceNet-Pytorch](https://github.com/timesler/facenet-pytorch)

[Faster R-CNN tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)