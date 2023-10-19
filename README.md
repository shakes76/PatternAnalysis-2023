# Visual Transformer for Classifying Alzheimer's Disease
Author: Tommy Wong, s4744424

## Visual Transformer Algorithm
The Visual Transformer (ViT) is a deep learning model that classifies objects in an image by taking advantage of the transformer encoder architecture, which has a multi-head self-attention mechanism, originally built for processing language-related tasks and finding relationships between words. The major novelty the ViT implements is the patch embedding of inputs; ViT divides images into fixed-sized patches, which allows the model to use a self-attention mechanism to build learnable relationships between pixels in an image, similar to how this is done for words in sentences. This implementation uses 12 heads for a multi-head attention mechanism, which allows more types of relationships to be built between pixels.

The dataset utilised is the ADNI dataset, consisting of brain MRI scans.

## The Problem
ViTs, with sufficient and fine-tuned training, are powerful for classifying images. This is particularly useful in medical imaging, such as MRI scans; it can help with assisting human classification of the MRI scans. This model will be used to classify MRI scans by one of two classes: Alzheimer's disease and normal cognition. 

## How it works
The architecture of the model was taken from the paper ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929)
![ViT architecture](https://viso.ai/wp-content/uploads/2021/09/vision-transformer-vit.png)

First, the image is split into fixed-sized patches. For this model, the input images are first resized to 224 by 224, then split into 16 by 16 patches.
Then, a learnable classification token is prepended to perform classification.
A positional embedding is also added to retain positional information of the pixels.
Then, these are put into the transformer encoder, which in this model, contains 12 transformer encoder layers, each consisting of a layer norm, followed by the multi-head self-attention layer, followed by another layer norm, followed by an MLP block.

Finally, an MLP head, consisting of a layer norm, followed by a linear layer, outputs the probability of each class. The index of the max of this tensor is taken as the classification.

The following outlines the architecture of the model and the shape of the inputs as it is processed:
1. Input: [8, 1, 224, 224] -> [batch size, channels, image height, image width] 
2. Patch embedding + positional embdding: [8, 197, 768] -> [batch size, number of patches + 1 (for the classification token), embed dimensions]
3. 12 Transformer encoder layers: [8, 197, 768]
4. MLP head classifier: [8, 2] -> [batch size, number of classes]

### Multihead Self-attention layer
The multihead self-attention layer allows different types of relationships between pixels to be learned. Any input is a query that is matched with the keys, which is related to a value.
The attention weights between two pixels are based on the similarity of their queries and keys. 
The attention function is given by:
Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V

### MLP block
The MLP block is a feedforward layer that consists of linear layers and activation functions. For this model, the MLP block uses a GELU activation function to introduce non-linearity to the model. These MLP blocks feeds into the next transformer encoder layer.

## Dependencies
This project requires the following libraries:

- torch==2.0.1
- torchvision==0.15.2
- matplotlib==3.7.2
- cuda==11.7

You can install torch, torchvision, and cuda on a conda environment with the following:
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```
Matplotlib:
```
conda install -c conda-forge matplotlib
```
Other versions of torch and matplotlib have not been tested.

## Data Preprocessing
The ADNI dataset comes in the 'train' and 'test' folders. A validation set was created from the training set using an approximately 80/20 split; the first new patient after 0.8 of the training set was set as the splitting point to maintain patient level splits.

All images were resized to 224 by 224 as per the seminar paper. All images were also transformed into grayscale since the MRI images are naturally grayscale.

## Results



