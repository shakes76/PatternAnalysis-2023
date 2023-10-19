# Alzheimer's Disease Classification using a Vision Transformer (ViT)
## Introduction
A Vision Transformer (ViT) was used to classify Alzheimer's disease from brain scans. The ADNI brain dataset was used.

## Model Architecture
The Vision Transformer from [] was used. Figure 1 illustrates the model architecture.

Initially, the PyTorch base ViT model was used (vit_b_16) with base weights (IMAGENET1K_V1).

Once working results were obtained and all other files finished, I reimplemented the ViT from scratch.

For instance, the provided ViT from pytorch has 3 output channels, where only 1 is needed (since all images are greyscale).

![](/recognition/ViT-ADNI-s4532823/assets/Model_architecture_Dosovitskiy.png)

## Data Loading & Preprocessing
### About the Dataset
The ADNI dataset was obtained from the COMP3710 Blackboard site, as well as from the Rangpur HPC.

The training set has 21,520 images total. 11,120 cognitive normal (NC) brain scans and 10,400 Alzheimer's disease (AD) scans.

The test set has 9,000 images total. 4,540 cognitive normal (NC) brain scans, and 4,460 Alzheimer's disease (AD) scans.

The dataset contains 20 scans for each patient. And obviously for a given patient all 20 images will be in the same directory (e.g. ./train/AD). 


### Data Loading
The dataset is loaded from different paths based on different contexts. 
+ On Linux (i.e. the Rangpur HPC), it's loaded from `/home/groups/comp3710/ADNI/AD_NC`
+ On Windows (i.e. my PC), it's loaded from `D:\ADNI\AD_NC`

The paths above can be altered as necessary in the `dataset.py` file.

A custom data loader was created to load the dataset, with the ability to split the training set into a training and validation set, with a default 80 train/20 val split. 

Additionally, one can randomly remove a certain percentage of the data from the data loader. This was used to speed up code execution when creating and debugging train.py. When this percentage is unspecified, none of the data is removed.


### Preprocessing
The training images preprocessed using the following data transforms (to combat overfitting and ensure data compatibility):
+ A RandomHorizontalFlip
+ A RandomVerticalFlip
+ A RandomCrop to a 224x224 image
+ The addition of 3 Grayscale channels
+ Conversion to a tensor

The validation and training images are preprocessed to a lesser extent (to ensure data compatibility):
+ A CentreCrop to a 224x224 image
+ The addition of 3 Grayscale channels
+ Conversion to a tensor

## Training & Results
### Training
The Vision Transformer model is trained for 10 epochs, criterion cross entropy loss, optimiser Adam, learning rate scheduler LRStep (halved every epoch). Initial learning rate 2e-5, patch size 16, 12 encoder layers, 2 outputs from the classification head (MLP layer). IMAGENET1K_V1 weights used, 3 greyscale input channels.

### Testing & Results


## Dependencies & Reproducability
### Dependencies
On Windows, Python 3.11.4 and PyTorch 2.0.1 within the Miniconda base environment are used for this implementation. 

On Linux, a Miniconda virtual environment is used, which also has Python 3.11.4 and PyTorch 2.0.1.

In addition, the packages `torchvision`, `pillow` and `matplotlib` are used within this implementation. All dependencies can be installed using the following:
```
pip install pytorch torchvision pillow pandas matplotlib
```

CUDA may also need to be installed on the system of use, to ensure this implementation can be run on 

### Running the Code
+ First change the working directory to `recognition/ViT-ADNI-s4532823`
+ Run `python train.py` in the terminal to train the model
+ Run `python predict.py` in the terminal to run predictions on a random image(s) from the dataset.

Ensure you are using the correct interpreter or have activated the correct virtual environment before running. 

## References