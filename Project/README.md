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
For the object detection, the aim is to use a YOLO v1 model to identify the location of faces and determine the gender of the person shown in the image.

For this task a YOLO v1 was implemented. Another three models using ResNet18-34-50 as a backbone were used to accomplish the task

Due to poor quality inference results, other models were explored.

- Faster R-CNN

A pretrained faster R-CNN model was fine tuned with the Female and Male images obtaining an Average Precision (AP) of `0.3434`


## Object recognition

- FaceNet (ResnetInception v1)
A pretrained resnetInception model was used to get a 512 dimensional embedding.
The distance between the resulting vectors was measured to 


## References

[FaceNet-Pytorch](https://github.com/timesler/facenet-pytorch)

[Faster R-CNN tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)