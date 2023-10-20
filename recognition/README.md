# YOLO: Object Detection and Classification
## Overview
The algorithm used in this project is YOLO. It is a one stage object detection algorithm. The architecture for this project consists of 12 convolution layers with batch normalisations in between. There are also some maxpooling layers at the beginning. This structure was based on the tiny-YOLO-v3 architecture shown below:
<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://www.researchgate.net/publication/338162578/figure/fig1/AS:839998031032320@1577282545408/The-network-structure-of-Tiny-YOLO-V3.jpg">
(image from reference 1)

The aim of this algorithm is to detect skin lesions and classify them as either melanoma, or seborrheic keratosis, with an intersection over uniou (IoU) score of 0.8. The datset used is the ISIC Dataset.

## How it Works
YOLO tries to draw a box around any detections in a given image. To do this it first takes the image and feeds it through the convolution layers to produce a feature map. A grid is then created and boxes of a preset size are generated for each square of the grid. This is donw be adding an offset to the initially generated boxes. Each box is then given an objectiveness score which represents the models confidence that there is an object in the box. The probabilities for each class that could be detected are recorded, as well as the box dimensions. The probabilities and objectiveness are found by decoding the feature map. The algorithm finally gives back a tensor containing all the boxes and their attributes.

The bounding boxes can then be filtered by their objectiveness score using non-maximum suppression. This just means the box with the highest objectiveness score is returned. The object in the box is determined to be the one with the highest probability. The box and object label can then be displayed on the image.

The image below shows the desired result of this YOLO algorithm using the ICIS Dataset:



## Dependencies
* Python
* torch
* numpy
* pandas
* time
* os
* cv2

## Pre-processing
When the images are loaded into 

## Usage and Results
### Usage
After the model has been trained the predict() function in predict.py can be called. To use it just pass the path to the desired image you wish to detect an object in. The function will perform pre-processing then display the image with the predicted boundary box and object name.

### Results
Whether a box is accurate or not can be determined by using the intersection over uniou (IoU) score. This determines how much of the predicted box overlaps with the true box. The average IoU score for this algorithm was determined to be 0.15 in the testing phase. This is very low. The main issue is believed to stem from the custom loss function. YOLO uses the squared error of three different terms in order to determine loss. The implementation of these formulas may be incorrect leading to the algorithm not being able to learn.

Here is an example output of the current algorithm:


## References
1.
2.
3.

