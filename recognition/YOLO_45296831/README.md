# Lesion Detect
A YOLO network to detect lesions within the [ISIC 2017/8 data set](https://challenge.isic-archive.com/data/#2017).
## Task
The ISIC 2017/8 data set contains colour images of over 2000 skin lesions to help AI researchers in discovering techniques to detect melenoma. The task of this report is to use a YOLO object detection network to locate the lesions in these images using segmentation data.
## Model
Image classification models have existed for a long time, with an example like [LeNet](https://en.wikipedia.org/wiki/LeNet) existing since 1990. Object detection is an extension of this, being able to localise multiple objects within an image.
![Image classification vs object detection](https://ambolt.io/wp-content/uploads/classification-object-detection.png)
YOLOv8, you only look once, is a comprehensive computer vision AI tool. It uses a one-stage single deep convolutional neural network architecture. It sees the entire image during training to make predictions of bounding boxes and class probabilities all at once, hence the name.
A YOLO model can be trained on a custom dataset to be suitable for a wide range of applications. Given an image plus an accompaning text file detailing the boundaries of objects to be detected, YOLO will provide a fast and accurate model for object detection.

## Training
Justify training, validation, and testing splits

## Results

## Usage
Download data with segmentation png
Run dataset.py on segmentation png folder by editing slurm.sh
Structure data
Run train.py

## Dependencies
[PyTorch](https://pytorch.org/)
[ultralytics](https://pypi.org/project/ultralytics/)
[opencv-python](https://pypi.org/project/opencv-python/)