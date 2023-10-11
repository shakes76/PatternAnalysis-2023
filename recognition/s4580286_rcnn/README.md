# Detecting lesions within the 2017 ISIC dataset with Mask RCNN 
**Shreya Singh:Task 3** 

**Student Number: s4580286**


This implementation trains a mask R-CNN to segment and classify images of skin lesions to help develop a diagnosis of melanoma or benignity from demoscopic images. This method is an extension of Fast RCNN with added pixel-pixel alignment and bounding box localisation and has been proven effective with datasets such as the COCO dataset, implementing both semantic and instance segmentation. Semantic segmentation classifies each pixel into a fixed number of classes, which differentiates similiar object instances. This is used to classify the mole from the background in this particular dataset. Instance segmentation, on the other hand, differentiates between each object instance, seperating each mole within this dataset. An illustration of the architecture is below.

<img width="515" alt="archetecture" src="https://github.com/Shreya-Personal/s4580286/assets/141000874/8a4876cf-1e1d-4f50-85f4-edd914125b68">

Figure 1: Mask R-CNN architecture

## Architecture
The backbone network used was the resnet50 fpn, which was pretrained on the COCO dataset. The backbone extracts hierarchical features from the input image. The Mask R-CNN then uses an RPN (region proposal network) to suggest potential bounding box locations before being ranked by the likelihood of containing objects. The ROI align allows for accurate, smooth pixel-level alignment between feature maps and region proposals, a signature feature of the Mask R-CNN. The aligned features are then passed through two parallel features for classifying and refining the segmentation. A third branch utilizes a series of convolutional layers to generate an instance-level binary mask. The loss function is multitask to evaluate classification, bounding box, and mask prediction during training.

## Metrics 
The chosen metrics are IoU and Accuracy. IoU is the Intersection over Union and measures the overlap between the predicted and groundtruth bounding boxes (Figure 2). Accuracy is classed as the number of correct identifications divided by the total amount images. This measures the accuracy of classification as either melanoma or benign. 

![image](https://github.com/Shreya-Personal/s4580286/assets/141000874/dc030b86-5fe0-4905-8496-cb156a20dad0)

Figure 2: IoU 

## Files & Dependancies
The files within this repository are modules.py,train.py, dataset.py, and predict.py. Modules.py contains methods to load the pretrained Mask R-CNN model and requires access to pytorch. Dataset.py imports and preprocesses images and files within the filepath, outputting transformed images and targets. The images are preprocessed using Torch and PIL libraries to turn them into normalised tensors before training. 'Targets' contains all ground truth data, including the bounding boxes, labels (for classification), and segmentation masks. Train.py inputs the filepaths to the images and groundtruths and outputs a binary torch model that can be loaded within predict.py. Predict.py samples the validation data and produces an accuracy figure along with a boxplot depicting the IOU, requiring matplotlib and torchvision's box_iou.

### Dependancies 
- tensorflow = 2.7.0
- matplotlib = 3.4.2
- PIL = 8.3.1
- pytorch = 2.1
- numpy = 1.20.3

## mask_rcnn_main.py 
To run, this code requires access to the 2017 ISIC dataset. This was downloaded from https://challenge.isic-archive.com/data/#2017 before beginning. This test driver script, when given the correct inputs for images, masks and diagnosis, will produce some example visualisations, an overall accuracy, and a boxplot of the IOU for the chosen validation data.

The training and testing data was pre-split; this was retained as the training and test set. Due to memory constraints, a random subset of 500 images was taken from the training data. The pretrained model used a batch size of 16 with a learning rate of 0.02 with stochastic gradient descent. As a smaller batch size of 2 (1/8 of the original) was taken due to memory constraints, the original learning rate was divided by 8 to get a learning rate of 0.0025, which was used for this implementation. 30 epochs were chosen, as after 30, the minima of loss was found (see visualisation below).

![image](https://github.com/Shreya-Personal/s4580286/assets/141000874/49f39e6d-9331-4244-a868-10db38b4789a)

Figure 3: Loss vs Epochs


## Performance
Below is a small sample to show the results. It shows the original image, the ground truth, and its predicted mask, label, and bounding box.

![Picture1](https://github.com/Shreya-Personal/s4580286/assets/141000874/41470911-97c3-476c-886d-a853dd343698)

Figure 4: Example of groundtruth vs prediction

The accuracy of the labels gave an accuracy of 66% for the classification of the mole as benign or melanoma. However, the IOU saw a median of 0.83, as shown in the boxplot below.

![Boxplot](https://github.com/Shreya-Personal/s4580286/assets/141000874/c0249bb4-30e8-435f-9128-d7fb5cae27b0)


Figure 5: IoU Boxplot



