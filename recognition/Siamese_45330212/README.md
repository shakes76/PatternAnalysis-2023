# Siamese Network Classifier based on ResNet-18
## Description
This Siamese Network-based classifier is designed for image similarity tasks. Leveraging a ResNet-18 architecture for feature extraction on the ADNI dataset for Alzheimer’s disease, the classifier produces embeddings for a pair of these images, which are then fed into a simple binary classifier. This model can be particularly useful in tasks where the goal is to determine if two images belong to the same class. Unlike a regular CNN classifier, which works to sort images into several classes, a Siamese Network essentially learns to become extremely adept at playing 'spot-the-difference' between two very similar images.
## How it Works
The Siamese Network I implemented uses a Triplet network. This network consists of three identical sub-networks that process three different images in parallel. Two of these images are of the same class, whilst the third is of the different class. The sub-networks are based on the ResNet-18 architecture and produce high-dimensional embeddings for the images by removing the final dense layer of the network. These embeddings are then compared and classified by a subsequent binary classifier constructed using 3 linear layers that use batch normalisation, relu activation functions, and dropout functions.

 
Figure 1 Siamese Triplet Model
(README_Images/image-1.png)
 
Figure 2 ResNet 18 Model (last FC layer is removed to produce detailed embedding)
(README_Images/image-2.png)

## Dependencies
Python 3.x
Torch 1.8.0
torchvision 0.9.0
NumPy
PIL
Matplotlib
Note on Reproducibility: To ensure reproducible results, set the random seed for both NumPy and PyTorch.

## Usage
 
Figure 3 Loss Plot of Siamese Network (y-axis loss, x-axis iterations over 20 epochs)
(README_Images/image-3.png)
 
Figure 4 Loss Plot for Classifier over 16 epochs
(README_Images/image-4.png)
 
Figure 5 Loss Plot for Classifier over 20 epochs
(README_Images/image-5.png)

From looking at the values of the loss over the different epochs it appears that the loss starts to remain consistent around the 15 epoch mark. Since the model is not significantly improving after this time, to prevent overfitting, I trimmed the number of epochs from 20 to 16.
Currently the classifier network classifies the test set with 71.27% accuracy but achieves 98% accuracy on the validation set. This is indicative of over-fitting of the model and possible data leakage during training.

## Data Preprocessing
The images are converted to the RGB space and normalized using a calculated set of mean and standard deviation values. The data is transformed into tensors using PyTorch's transforms library. To generalise the data further, a random crop dataset is concatonated to the original dataset.

### Data Splits
The total dataset is divided into 70% training and 30% testing sets. This training set it then split into 80% used for training and 20% used for validation. This split ensures that the model is trained on a diverse set of data and generalizes well to unseen data.

## Code Comments
For detailed comments, please refer to the inline comments within the code.