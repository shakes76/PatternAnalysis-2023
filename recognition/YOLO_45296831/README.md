# Lesion Detect
A YOLO network to detect lesions within the [ISIC 2017/8 data set](https://challenge.isic-archive.com/data/#2017).
## Task
The ISIC 2017/8 data set contains colour images of over 2000 skin lesions to help AI researchers in discovering techniques to detect melenoma. The task of this report is to use a YOLO object detection network to locate the lesions in these images using segmentation data.
## Model
Image classification models have existed for a long time, with an example like [LeNet](https://en.wikipedia.org/wiki/LeNet) existing since 1990. Object detection is an extension of this, being able to localise multiple objects within an image.
![Image classification vs object detection](https://ambolt.io/wp-content/uploads/classification-object-detection.png)\
YOLOv8, you only look once, is a comprehensive computer vision AI tool. It uses a one-stage single deep convolutional neural network architecture. It sees the entire image during training to make predictions of bounding boxes and class probabilities all at once, hence the name.
A YOLO model can be trained on a custom dataset to be suitable for a wide range of applications. Given an image plus an accompaning text file detailing the boundaries of objects to be detected, YOLO will provide a fast and accurate model for object detection.

## Training
(Use the yaml file to see how it trains)
For this model, an 80/10/10 split for training, validating and testing data repectively was chosen.
Justify training, validation, and testing splits

## Results

## Usage
1. To run this model on your own device, you will first need access to the ISIC 2017/8 data set. You will need both the training input data and the training ground truth segmentation data.
2. Then you will need to configure the data paths on dataset.py so that it grabs the segmentation data and converts it to the text files that will be readable from YOLO.
3. Next, you will have to structure you data in the following way, where images contains the training input JPEG's, and labels contains the text files you created from dataset.py\
<pre>
├── train
│   ├── images
│   │   ├── ISIC_0000000.jpg
│   │   ├── ISIC_0000001.jpg
│   │   ├── ISIC_0000002.jpg
│   │   ├── ...
│   └── labels
│       ├── ISIC_0000000.txt
│       ├── ISIC_0000001.txt
│       ├── ISIC_0000002.txt
│       ├── ...
├── val
│   ├── images
│   │   ├── ISIC_xxxxxxx.jpg
│   │   ├── ISIC_xxxxxxx.jpg
│   │   ├── ISIC_xxxxxxx.jpg
│   │   ├── ...
│   └── labels
│       ├── ISIC_xxxxxxx.txt
│       ├── ISIC_xxxxxxx.txt
│       ├── ISIC_xxxxxxx.txt
│       ├── ...
├── test
│   ├── images
│   │   ├── ISIC_xxxxxxx.jpg
│   │   ├── ISIC_xxxxxxx.jpg
│   │   ├── ISIC_xxxxxxx.jpg
│   │   ├── ...
│   └── labels
│       ├── ISIC_xxxxxxx.txt
│       ├── ISIC_xxxxxxx.txt
│       ├── ISIC_xxxxxxx.txt
│       ├── ...
</pre>
4. Finally, you can edit the .yaml file so that the train, test, and val directories match those that you created in the previous step, then you can configure then run train.py to train the model. Once train.py is finished, it will automatically create graphs and model details for you.

## Dependencies
[PyTorch](https://pytorch.org/)\
[ultralytics](https://pypi.org/project/ultralytics/)\
[opencv-python](https://pypi.org/project/opencv-python/)