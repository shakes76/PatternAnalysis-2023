# Recognition Tasks
Various recognition tasks solved in deep learning frameworks.

Tasks may include:
* Image Segmentation
* Object detection
* Graph node classification
* Image super resolution
* Disease classification
* Generative modelling with StyleGAN and Stable Diffusion
## Using Siamese network to classify Alzheimer’s disease (normal and AD) 

## Author
Name: Yicheng Jia

Student number: 46364184

Email: s4636418@student.uq.edu.au

This project was completed for COMP3710 report 2023.

# Decsription:

## Siamese Network for Alzheimer's Disease Classification

This project uses a Siamese Network to classify Alzheimer's disease based on image pairs.

The aim of this project is to create a classifier based on Siamese network to classify Alzheimer's disease (normal and AD), on ADNI brain data set(https://adni.loni.usc.edu/), and achieve a 0.8 accuracy.

The Siamese network is a special type of neural network architecture aimed at solving problems related to similarity comparison and image validation. 

In this architecture, two identical subnetworks (or two siamese subnetworks) will work in parallel, each receiving different inputs and outputting feature vectors.
 
These feature vectors are then joined or combined for further processing or comparison.

Then we can connect or combine these feature vectors for further processing or comparison, one example is to calculate the difference between the two outputting feature vectors, which can be represented as the similarity of the two inputs.

Values which are close to zero representing high similarity while values which are close to one representing high difference.


# Siamese Networks

## How They Work

1. **Input Stage**: Two different inputs go through two identical subnetworks (having the same parameters and weights).
2. **Feature Extraction**: Each subnetwork extracts features from its input.
3. **Distance Metric**: The extracted features are compared using some form of distance function like Euclidean distance or cosine similarity.
4. **Output**: The network outputs a similarity score between these two inputs.


## Applications

1. **Face Verification**: To determine if two given face images belong to the same person. 
This is often used by security departments to determine whether a person belongs to a specific group of people (such as wanted criminals or other important individuals)

2. **Image Verification**: To find the image in a database that is most similar to a given image. 
This is often used for image recognition and the Internet of Things, such as identifying the specific brand and model of a certain camera

3. **Signature/Fingerprint Verification**: To check if two signatures/fingerprints are from the same person.
Similar to facial recognition, signature and fingerprint recognition are also used to determine whether a handwriting/fingerprint belongs to a specific person, and are commonly used for criminal evidence correction


## Advantages

1. **Connected Network**: Both subnetworks will share model parameters, reducing the total number of model parameters.This saves a lot of computational resources and time compared to training two "disconnected" networks.

2. **Training Efficiency**: Due to parameter sharing,Siamese neural networks can converge faster than normal neural networks.

3. **Flexibility**: Although Siamese neural neural networks are mainly used for one-to-one comparisons, they can also be easily extended to one-to-many or many-to-many comparisons.


## Disadvantages

1. **Data Imbalance**: For example, a certain type of sample may be much smaller than other samples, which is common when comparing certain information (such as face/fingerprint) with other information in the database.

2. **Lack generalization ability**: Obviously, Siamese neural networks can only be used to handle certain specific problems and lack generalization ability.


## Resnet18 architecture

I used Resnet18 for this task, which is a convolutional neural network that is 18 layers deep. 

The Resnet18 is a pre-trained model, which is trained on the ImageNet dataset and used to extract features from the images. 

Further introduction of Resnet18:
https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/resnet-18-pytorch/README.md

For this project, I used Resnet18 to extract features from the two input images, and then calculate the distance between them, then, use the distance to classify the images.

The distance can be regarded as "similarity" of the two input images, closing to 0 means high similarity while closing to 1 means low similarity.


## Dataset
The dataset used in this project is ADNI brain data set, which is a public dataset. 

The dataset can be downloaded from [ADNI](http://adni.loni.usc.edu/data-samples/access-data/).

The dataset contains 4,424 images of brain MRI scans, which are divided into 2,315 AD and 2,109 NC images.

Here are several examples of AD (left) and NC (right) images:

![AD_example.jpeg](../Images/AD_example.jpeg)
![NC_example.jpeg](../Images/NC_example.jpeg)

The dataset have been preprocessed, they are all in same size (256 * 240 pixels) and in gray scale.

And all the images have the same naming format, which is "firstPart_secondPart.jpeg", the first part is the patient's unique ID, 

and the second part is the image's unique ID, each patient have 12 brain MRI scans.


# Files Description:

1. **modules.py**: 
    - Contains the Siamese Network architecture.
    - Uses a no pre-trained ResNet18 as the backbone.
    - Defines the forward pass for single images and image pairs.

2. **dataset.py**: 
    - Defines the Siamese Network dataset.
    - Loads image pairs and labels.
    - Includes data augmentation and preprocessing steps.

3. **train.py**: 
    - Contains the training/validating/testing loop for the Siamese Network.
    - Save the best module.
    - Logs training loss to TensorBoard.

4. **predict.py**: 
    - Uses the trained model to make predictions on the test set.
    - Calculates and prints the accuracy of the model.
    - Logs test accuracy to TensorBoard.


## Training/Validating/Testing:

In train.py, I defined three loops: Train, Validate and Test.

In each epoch, all of the three loops will run one by one.

This is because I can view the result once the codes are running, which allows me to modify parameters and stop early in some cases.


## Predicting/Recurrenting:

Since I defined the test loop in train.py, I ignore the testing progress in predict.py. 

However, I can load the saved best module and use it for recurrenting, proving we can get the result again.


## Main Components
Siamese Network (modules.py)
    - The Siamese network is a neural network designed to determine if two input images are similar. It uses ResNet18 as the base model and adds several fully connected layers on top.

Dataset (dataset.py)
    - The SiameseNetworkDataset class is used to load image data. It can load images from two categories (AD and NC) and generates a label for each pair of images indicating if they belong to the same category.

Training/Validating/Testing Loop (train.py)
    - This script contains the main training, validation, and testing loops for the Siamese network. 
    - It includes functions for checking CUDA availability, initializing datasets and dataloaders, defining the loss function and optimizer, and the main training loop with progress bars for training, validation, and testing.

Recurrent (predict.py)
    - This script will use the best module to recurrent the best result. It can also load data and visualize via tensorboard.

## How to Run:

1. Ensure you have all the required libraries installed.
2. Make sure the path of the dataset we need has been set correctly, the folder will be named "AD_NC" and be placed together with all of the py files. 
3. Run the scripts in the following order:
    - `train.py`: This will train the model and save the best weights.
    - `predict.py`: This will use the best trained model to recurrent the best result on the test set.
4. After running the `train.py`, your AD_NC directory structure should be as follows:


AD_NC/

|-- test/

|   |-- AD/

|   |-- NC/

|-- train/

|   |-- AD/

|   |-- NC/

|-- val/

|   |-- AD/

|   |-- NC/

## Notes
    - In the SiameseNetworkDataset class of dataset.py, some data augmentation code is commented out. 
    - If you wish to enhance the model's generalization capabilities, consider uncommenting this code.
    - Ensure the directory structure for the image data is correct when using the dataset class.
    - You must run train.py first to get the best module then run predict.py, otherwise there will be an error.


## Dependencies

### Scripts
- Python==3.11.4
- Pytorch==2.0.1
- Tensorboard==2.6.0

Run the following command to install the required packages, note that some other libraries such as 'os' not listed here.

```bash
pip install -r python==3.11.4
pip install -r pytorch==2.0.1
pip install -r tensorboard==2.6.0
```


### Hardware
This model have be trained 30 epochs on a RTX2080 for about 6 hours. Please take this as a reference and adjust your batch size with your hardware. 

For higher performance, recommend using higher GPU with higher memory.


# Reference
https://github.com/pytorch/examples/tree/main/siamese_network
https://www.kaggle.com/code/jiangstein/a-very-simple-siamese-network-in-pytorch
https://stackoverflow.com/questions/53803889/siamese-neural-network-in-pytorch

