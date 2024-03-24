## Covid Lung classification and Segmentation

The following sub-repository contains the information for the second part of the Deep Learning course examination.
The problem to be solved is the classification of X-ray images of healthy patients, patients with covid and patiens with a differnt pulmonary disease, expressed as non-covid on this problem.
The second task is semantic segmentation of the same group of images to classify the regions where the lungs are located.

For this problem, two approximations were tested.

- `First approximation`:
For the first approximation the usage of a graph was used to find a solution to the two problems at the same time. To create this graph, two pretrained MobileNet V3 neural networks and a pretrained U-Net were used.

- `Second approximation`
For ther second approximation two joined ResNet neural networks were trained and a U-Net was later trained in a different stage. The two trained graphs were joined togheter to solve both problems simultaneously.

The code on this repository can be found on the following [google drive folder](https://drive.google.com/drive/folders/1QL9J2Mm0WOoghS9Gy42QUsd3s-A84F_d?usp=sharing) were the code was hosted to get advantage of the GPU virtual machines.


Even though some of the files and folders on the google drive folder have different names than the ones hosted righ here, the file structure remains the same. In the following section the names of the google drive folders will be specificed inside paranthesis: 


```
- classification_segmentation/
    ├── TrainCovidNNGraph.ipynb
    ├── TestCovidNNGraph.ipynb
    ├── custom_load.py
    ├── fit_func.py
    └── unet_mobilenet.py
- classification_problem/
    ├── CovidClassificationTrain.ipynb
    |── CovidClassificationTest.ipynb
    ├── custom_load_resnet.py
    |── data_loading_resnet.py
    ├── fit_func_resnet_dual.py
    └── resnet18_dual_model.py
- segmentation_problem/
    ├── TestCovidSegmentation.ipynb
    ├── TrainCovidSegmentation.ipynb
    ├── custom_load.py
    ├── fit_func_segmentation.py
    └── unet_model.py
- joint_problem_pretrained/
    ├── TestCovidSegmentationClassification.ipynb
    ├── custom_load.py
    ├── resnet18_dual_model.py
    ├── unet_model.py
    └── unet_resnet.py
```

The first approximation code is located in the `classification_segmentation` folder, including the training and testing notebooks and the related scripts.
For the second approximation, the training and testing process of the classification is positioned in `classification_problem`, while the related files for the segmentation are placed in `segmentation_problem`.
Finally, the testing files for the joint pretrained models are located in `join_problem_pretrained`.

Results:

For the segmentation problem, the Jaccard Index or Intersecction Over Union metric was used, and Multiclass Accuracy and Multiclass PR-AUC were used to test the classification model.

| Approximation        | IOU    | Accuracy | PR-AUC  |
|----------------------|--------|----------|---------|
| Joint graph          | 0.9009 | 0.3685   | 0.3333  |
| Joint graph          | 0.8564 | 0.4942   | 0.3364  |
| Pretrained models    | 0.9443 | 0.5074   | 0.3352  |