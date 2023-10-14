# Pattern recognition Report, project ISIC2017/8 improved uNet

Project demonstrates segmenting of the ISIC 2017/8 deata set using the improved UNet, 
aiming for a minimum DSC of 0.8 on the test set

Contains:
* modules.py: 
Contains network architecture, and deep learning models used in the segmentation.

The improved uNEt module, along with subclasses such as context module, localization module and upsamling module is 
based on the structure in: https://arxiv.org/pdf/1802.10508v1.pdf


* dataset.py:
Class for loading and preprossesing the image-data 

* train.py:
Program for training the model
Training parameters based entirely on structure in https://arxiv.org/pdf/1802.10508v1.pdf
These are still not tested on the dataset, but will for now be temporarily used until a result is gotten 

* utils.py
Diceloss function, gotten from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
Currently does not seem to work correctly
