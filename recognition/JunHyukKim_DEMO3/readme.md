# JunHyuk Kim 2023 Report-ISIC 2017 improved Unet segmentation task.

This is a brief description of my project. The project is made for segmentation of melanoma. 
Task finished was "

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Description](#Description)


## Installation
1. Clone the repository
2. Download Dependencies
3. Download ISIC datasets.  The dataset should be positioned in the same folder as the python codes under ISIC Folder. The images should be positioned under the ISIC Folder, ISIC-2017_Training_Data folder, and another ISIC-2017_Training_Data folder.
4. Run the Train.py with python3 this will create the model.pth file. 
5. run Predict.py to get the segmentation and the dice coefficient. 
## Usage
Run the train.py to train the model and create a model.pth file. 
Using this file you can get the segmented image.


## Description

1. The readme file should contain a title, a description of the algorithm and the problem that it solves
(approximately a paragraph), how it works in a paragraph and a figure/visualisation.
The 

2. It should also list any dependencies required, including versions and address reproduciblility of results,
if applicable.
import torchvision
import torch
import matplotlib
import sklearn
import glob 
import PIL
import pandas
import numpy
python 3.11.5
window 11
the training dataset requires to be in data folder. The ground truth files 

3. provide example inputs, outputs and plots of your algorithm


4. The read me file should be properly formatted using GitHub markdown

5. Describe any specific pre-processing you have used with references if any. Justify your training, 
I used transformation to increase the number of samples the model is trained on.
I also resized the images into 256x256 sizes to increase the performance of training and also because my model works on that sizes. 

validation
and testing splits of the data.


1. description and explanation of the working principles of the algorithm implemented and the problem it
solves (5 Marks)
This program uses improved Unet to segment the differnt types of skin conditions such as melanoma.
The improved Unet strucuture explained
The improved Unet works by first, applying the 3x3 convolutional layer. After that context layer is applied. 
Inbewteen each context layer, 3x3 convolutional layer with 2 stride is used to reduce the size of features. 
After applying context layer 5 times and the stride 2 convolutional layer 4 times, the upsampling module is used and reduces the number of features, and increasee the size of each images. 
Segmentation layer is used and 
finally the 3x3 convolutional layer is used as the final convolutional layer. After this, final segmentaion layer is applied and it is combined with the upscaled segmention layer. 
This is ran though the activation function which is sigmoid in this case. 

By doing this, each features are extended and this can reliably create the segmentation.

![My Image Alt Text](image.png)
    
2. description of usage and comments throughout scripts (3 Marks)


3. proper formatting using GitHub markdown (2 Mark)




John Kim DEMO3
Task is: