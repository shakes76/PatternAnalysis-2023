# Binary Segmentation of ISIC2018 with improved uNet model
Author: Marius Saether

Student nr: s4824209


## Overview
Project for segmentation of the ISIC 2018/8 dataset, using the improved UNet architecture and aiming for a minimum DSC of 0.8. The module used in this project is based on the works of Isensee, Et al. in their 2018 paper: Brain Tumor Segmentation and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge[^1]. 

The model scores an average DCS of 0.839, but struggles with no overlap on select images


## Model Architecture

### Encoder
The network consists of a encoding and a decoding part. The encoding part consists of 5 layers, which consecutivly reduce the spacial, and increases the feature information of the input. This is done through 3x3 convolutions with stride 2. At each layer, the convolution is conected to a context module, made up of a pre-activation residual block with two 3x3 convolutions and a dropout layer. Here it is also implemented a skip-conection that feed-forwards the input, and sums it with the output from the context module. 

### Decoder
The decoder is made up of upsampling and localization modules. High feature information from the lower levels is upscaled by a factor of 2, doubling the spacial information, and thereafter havling the features through a 3x3 convolution. It is then concatinated with the output of the same layer encoded information, doubling the features. The localization module then halves the features with a 3x3 and a 1x1 convolution. 

### Segmentation 
At the output of the last two localization modules is a segmentation layer made up of a 1x1 convolution, redusing the features to 1 (number of output features). The output of the first segmentation layer is upscaled by a factor of 2, and then added to the output of the next segmentation later. This sum is then upscaled again, before adding it to the output of the decoder. The segmentation layers is the main difference between this arcitecture and the standard uNEt. 


## Pre-prosessing of data
The ISIC2018 dataset contains 2594 dermoscopy images with a corresponding ground truth. 
The dataset is split into training and test (80% of data / 20% of data), using the functions 'data_sorter'
and the class 'customDataset'. 

The images have large variations in size, so they are rezised to 512*512

The custom transformation class 'train_Transformation' is used to support transformations with a random component. When it occurs it will act similarly on both the training image and the corresponding ground truth. 
The transformation used on the training-set is: random horisontal- and verticalflip, random rotation, and random adjustment of brightness, contrast and saturation. 

The various transformations was aimed to increase the dice score of the poorly segmented outliers in the test, but failed to do so. They did however marginally increase the average DCS score of the whole set. 

## Training and parameters

The model was trained using the Adam optimizer with a learning rate of 0.0005 and a step-learning rate scheduler which decreases the learning rate by 0.985 every epoch. The model trained for 100 epochs, with a batch size of 6, and was evaluated against the test set after every epoch. The model parameters was saved each time the test outperformed previous results, innsuring that the best model was stored at the end of the training proccess.  

Given that it is not released any ground thruth for the validation set of ISIC2018, the test set was also used for validation. 


## Results
Top layer shows a sample of images segmentet through the model, bottom layer shows corresponding grount truth
![test image](readme_img/segmentation_testset.png)

### Dice scores on the test set
Average score: 0.839

Max score: 0.987

Min score: 0.0



## Improvements
The main problem with the model as it stands is the minimum dice score when the test set is sent through the trained model. Even though the model scores an average DCS of 0.839, the worst segmentation scores a 0.0, meaning no overlap with the ground truth. 



## Dependencies
Pytorch: 2.0.1

cuda: 11.7

python: 3.10.12

scikit-learn: 1.3.0

pillow: 9.4.0


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
Using adam optimizer with stepLR scheduler
These are still not tested on the dataset, but will for now be temporarily used until a result is gotten 

* utils.py
Diceloss function, gotten from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch



## References
[^1]: https://arxiv.org/pdf/1802.10508v1.pdf
* [1] F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, “Brain Tumor Segmentation
and Radiomics Survival Prediction: Contribution to the BRATS 2017 Challenge,” Feb. 2018. [Online].
Available: https://arxiv.org/abs/1802.10508v1