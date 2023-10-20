# Siamese Network Classifier based on ResNet-18
## Description
This Siamese Network-based classifier is designed for image similarity tasks. Leveraging a ResNet-18 architecture for feature extraction on the ADNI dataset for Alzheimer’s disease, the classifier produces embeddings for a pair of these images, which are then fed into a simple binary classifier. This model can be particularly useful in tasks where the goal is to determine if two images belong to the same class.

## How it Works
The Siamese Network consists of two identical sub-networks that process two different images in parallel. The sub-networks are based on the ResNet-18 architecture and produce high-dimensional embeddings for the images. These embeddings are then compared and classified by a subsequent binary classifier to indicate if the images are similar or dissimilar.

![https://medium.com/swlh/one-shot-learning-with-siamese-network-1c7404c35fda](image-1.png)
![https://www.researchgate.net/figure/Original-ResNet-18-Architecture_fig1_336642248](image-2.png)

## Dependencies
Python 3.x
Torch 1.8.0
torchvision 0.9.0
NumPy
PIL
Matplotlib
Note on Reproducibility: To ensure reproducible results, set the random seed for both NumPy and PyTorch.

## Usage
Loss Plot (y-axis loss, x-axis iterations over 1 epoch)
![Alt text](image.png)

## Data Preprocessing
The images are converted to the RGB space and normalized using a calculated set of mean and standard deviation values. The data is transformed into tensors using PyTorch's transforms library.

### Data Splits
The dataset is divided into 70% training and 30% testing sets. This split ensures that the model is trained on a diverse set of data and generalizes well to unseen data.

## Code Comments
For detailed comments, please refer to the inline comments within the code.