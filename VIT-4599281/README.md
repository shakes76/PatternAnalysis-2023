# Pattern Analysis
Pattern Analysis of various datasets by COMP3710 students at the University of Queensland.

We create pattern recognition and image processing library for Tensorflow (TF), PyTorch or JAX.

This library is created and maintained by The University of Queensland [COMP3710](https://my.uq.edu.au/programs-courses/course.html?course_code=comp3710) students.

The library includes the following implemented in Tensorflow:
* fractals 
* recognition problems

In the recognition folder, you will find many recognition problems solved including:
* OASIS brain segmentation
* Classification
etc.

# Classification for Alzheimer's disease of the ADNI brain data

## Description for the Alzheimer's disease of the ADNI brain data

## Description of the algorithm

## Problem this project solves
This project utilises a modified dataset sourced from [ADXI dataset for Alzheimer's disease](https://adni.loni.usc.edu/). The dataset contains two rerositories which are AD(disease) and NC(normal) and each repositorycontains training and test repositories. This project aims to utilise the Vision Transformer to classify the health brain images and unhelath images.

## How it works
### Documents detail
│  dataset.py  
│  LICENSE  
│  modules.py  
│  predict.py  
│  processing.py  
│  README.md  
│  train.py  
│     
├─config  
│  │  config.py  
│  │  
│  └─__pycache__  
├─model  
│  │  resnet.py  
│  │  vit.py  
│  │  __init__.py  
│  │  
│  └─__pycache__  
└─utils  
    │  utilis_.py  
    │  utilis_augs.py  
    │  utils_fit.py  
    │  utils_loss.py  
    │  utils_model.py  
    │  __init__.py  
    │  
    └─__pycache__  

Function of each document:
* processing.py  
This document helps segment the training set to training and validation set by patient ID. The training set contains 80% patients of the original dataset and the validation set contains the other 20%. In addition, it also changes the storing format to the same format as the ImageNet. 
Example:    
Before:      
├─dataset  
│  ├─train  
│  │  ├─AD  
│  │  └─NC  
│  └─test  
│     ├─AD  
│     └─NC  
After:      
├─dataset   
│  ├─train             // 80% of the orginal train set  
│  │  ├─0  
│  │  └─1  
│  ├─val               // 20% of the orginal train set  
│  │  ├─0  
│  │  └─1  
│  ├─test  
│  │  ├─0  
│  │  └─1  
│  └─ label.txt        // store the label information  

* dataset.py  
This document provides the image transformation for the training, validatioin and test data and provides data augmentation for the training data. In addition, this document also provides the built in dataloader functioin to load these three data set.  
  
* modules.py  
This document use the `select_model` function in `utils/utils_model` to select the model that set up in the `opt.model_name`. To be specific, the vit model store in the `model/vit.py`. By utilising this structure, the user can add different models to the repository `model` and utilise them in the future.  
Note:  
When add new model to the repository `model`, the user should import the added document in `model/__init__.py`.  
Example: if the user add a resnet to the model they should add the following code to the `model/__init__.py`.  
```
from .resnet import *
```
  
* train.py  
This document is the main document for training and validating purpose. It utilises the `dataset.py` to transform and load the images and utilise the `modules.py` to load the vit model. In addition, this document provides various functions including `resume`, `mix precision`, `warm up`, `early stopping` and `learning rate scheduler`.  
This document utilises the `fitting` function in `utils/utils_fit` for the training and evaluation purpose. For the training phase, this document updates the *loss*, *optimizer* on each batch size. For the evaluation phase, this document caculate the validating loss and accuracy on each epoch.  
During the training and validating procedure, this document also provides the information of the *training loss*, *validating loss*,*learning rate*, *train acc* and *validating acc* etc on each epoch. In addition, it stores the best parameters of the model on the given `opt.save_path`.  
  
* predict.py  


### Dependencies required
python 3.9  
pytorch 2.0.1  
torchvision  
tqdm  
numpy  
scikit-learn  
matplotlib  
prettytable  
grad-cam  
opencv-python  
pillow  

Installation Example
```
pip install grad-cam
```
### How to run  
  
Train and Validate  
```
python train.py --config config/config.py --save_path runs/vit --lr 1e-4 --warmup --amp  --batch_size 128 --epoch 300
```

Predict
```
python predict.py --task test --save_path runs/vit --visual --tsne
```
