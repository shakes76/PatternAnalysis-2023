# YOLO: Object Detection and Classification
## Overview
This project aims to build a YOLO object detector and classifier. The base structure of the model is base on the YOLO-tiny-v3 architecture. 

This project uses the ISIC dataset and provides a custom datset class in order to load the dataset for the model.

The implementation of the custom loss function was difficult as my understanding of the three parts of the formula is lacking. This function is something I wish to improve upon.

Another thing that I was to implement is the use of route layers in the model architecture. This would reduce the degredation problem that happens with deeper netwroks, i.e. the accuracy should improve.

## Principles of YOLO
YOLO uses a base network called darknet, which is a CNN with a lot of convulution layers, to extract features from a given image. Pre-set bounding boxes are then used as a base to generate the following information:
* X centre coordinate of the box, calculated from base box coordinate and an offset
* Y centre coordinate of the box, calculated from base box coordinate and an offset
* Width of box
* Height of box
* Probability that an object is within the box
* Probability for each class the model is built to detect

From here, the generated boxes can be filtered by using the probabilities of the classes to determine if there is an object and, if there is, what object it is.

There will be multiple positive boxes. These can be wittled down using Non-Maximum Suppression. The accuracy of the predicted box can be evaluated by using IoU.

### IoU

### Non-Maximum Suppresion

## Loss Function

