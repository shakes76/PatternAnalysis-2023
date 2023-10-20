# YOLO: Object Detection and Classification
## Overview
The algorithm used in this project is YOLO. It is a one stage object detection algorithm. The architecture for this project consists of 12 convolution layers with batch normalisations in between. There are also some maxpooling layers at the beginning. This structure was based on the tiny-YOLO-v3 architecture shown below:
![The-network-structure-of-Tiny-YOLO-V3](https://github.com/LazyScribble/PatternAnalysis-2023/assets/141600341/487de0ff-dc86-48df-b19a-aa1b55a35897)

(image from reference 1)

The aim of this algorithm is to detect skin lesions and classify them as either melanoma, or seborrheic keratosis, with an intersection over uniou (IoU) score of 0.8. The datset used is the ISIC 2017 Dataset, refer to reference 4 for more information.

## How it Works
YOLO tries to draw a box around any detections in a given image. To do this it first takes the image and feeds it through the convolution layers to produce a feature map. A grid is then created and boxes of a preset size are generated for each square of the grid. This is donw be adding an offset to the initially generated boxes. Each box is then given an objectiveness score which represents the models confidence that there is an object in the box. The probabilities for each class that could be detected are recorded, as well as the box dimensions. The probabilities and objectiveness are found by decoding the feature map. The algorithm finally gives back a tensor containing all the boxes and their attributes.

The bounding boxes can then be filtered by their objectiveness score using non-maximum suppression. This just means the box with the highest objectiveness score is returned. The object in the box is determined to be the one with the highest probability. The box and object label can then be displayed on the image.

The image below shows the desired result of this YOLO algorithm using the ICIS Dataset:
![desired_result](https://github.com/LazyScribble/PatternAnalysis-2023/assets/141600341/9a33f633-3446-4876-a567-02f1ed311611)

## Dependencies
* Python 3.10.12
* torch 2.1.0+cu118
* numpy 1.23.5
* pandas 1.5.3
* cv2 4.8.0

## Pre-processing
When the images are loaded into the dataset class they are resized to 416x416 as the algorithm expects this size. They are also normalised by dividing by 255. This pre-processing is also done when the predict() dunction is called.
The train, test and validation data for the ISCI dataset is already split on their website and must be downloaded seperately. 

## Usage and Results
### Usage
Change all the image, mask and checkpoint paths in train.py and predict.py to suit you. Then if there is no checkpoint available, run train.py. Then run predict.py with the path to the image you wish to predict.

Here is an example input and output of the current algorithm:
```python
checkpoint_path = /content/drive/MyDrive/Uni/COMP3710/checkpoint"
image_path = "/content/drive/MyDrive/Uni/COMP3710/ISIC-2017_Training_Data/ISIC_0000004.jpg"
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
predict(image_path, model)
```

![predict_image](https://github.com/LazyScribble/PatternAnalysis-2023/assets/141600341/7be53da0-7473-403f-817d-6a417ca70eda)

### Results
Whether a box is accurate or not can be determined by using the intersection over uniou (IoU) score. This determines how much of the predicted box overlaps with the true box. The average IoU score for this algorithm was determined to be 0.47 in the testing phase. This is lower than the desired outcome of 0.80. The main issue is believed to stem from the custom loss function. YOLO uses the squared error of three different terms in order to determine loss. The implementation of these formulas may be incorrect leading to the lack of improvement in the train step. 

## References
1. Fang, Wei & Wang, Lin & Ren, Peiming. (2019). Tinier-YOLO: A Real-time Object Detection Method for Constrained Environments. IEEE Access. PP. 1-1. 10.1109/ACCESS.2019.2961959. https://www.researchgate.net/figure/The-network-structure-of-Tiny-YOLO-V3_fig1_338162578
2. Kathuria, A. (2020, December 15). How to implement a Yolo (V3) object detector from scratch in pytorch: Part 3. Paperspace Blog. https://blog.paperspace.com/how-to-implement-a-yolo-v3-object-detector-from-scratch-in-pytorch-part-3/ 
3. Hui, J. (2022, September 6). Real-time object detection with Yolo, yolov2 and now yolov3. Medium. https://jonathan-hui.medium.com/real-time-object-detection-with-yolo-yolov2-28b1b93e2088#:~:text=YOLO%20uses%20sum%2Dsquared%20error,box%20and%20the%20ground%20truth).
4. Isic Challenge. (n.d.). https://challenge.isic-archive.com/data/#2017 


