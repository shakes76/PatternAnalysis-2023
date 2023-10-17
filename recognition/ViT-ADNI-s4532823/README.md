# Alzheimer's Disease Classification using a Vision Transformer (ViT)
## Introduction
A Vision Transformer (ViT) was used to classify Alzheimer's disease from brain scans. The ADNI brain dataset was used.

## Model Architecture
The Vision Transformer from [] was used. Figure 1 illustrates the model architecture.

Initially, the PyTorch base ViT model was used (vit_b_16) with base weights (IMAGENET1K_V1).

Once working results were obtained and all other files finished, I reimplemented the ViT from scratch.

For instance, the provided ViT from pytorch has 3 output channels, where only 1 is needed (since all images are greyscale).

## Data Loading & Preprocessing
### Dataset
The ADNI dataset was obtained from the COMP3710 Blackboard site, as well as from the UQ Rangpur HPC.

The training set has 21,520 images total. 11,120 cognitive normal (NC) brain scans and 10,400 Alzheimer's disease (AD) scans.

The test set has 9,000 images total.

The dataset contains 20 scans for each patient. And obviously for a given patient all 20 images will be in the same directory (e.g. train/AD). 

### Data Loading
A custom data loader was created to load the dataset, with the ability to split the training set into a training and validation set, with a default 80 train/20 val split. 

Additionally, one can randomly remove a certain percentage of the data from the data loader. This was used to speed up code execution when creating and debugging train.py. By default, none of the data will be removed from the data loader.

### Preprocessing
The training 

## Training & Results
### Training

### Results

## Dependencies & Reproducability
### Dependencies
```
install pytorch torchvision pillow pandas matplotlib
```

### Reproducability

## References