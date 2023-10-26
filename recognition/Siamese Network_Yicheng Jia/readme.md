# Siamese Network for Alzheimer's Disease Classification

This project uses a Siamese Network to classify Alzheimer's disease based on image pairs.

## Files Description:

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
      
## Main Components
Siamese Network (modules.py)
The Siamese network is a neural network designed to determine if two input images are similar. It uses ResNet18 as the base model and adds several fully connected layers on top.

Dataset (dataset.py)
The SiameseNetworkDataset class is used to load image data. It can load images from two categories (AD and NC) and generates a label for each pair of images indicating if they belong to the same category.

Training Loop (train.py)
This script contains the main training, validation, and testing loops for the Siamese network. It includes functions for checking CUDA availability, initializing datasets and dataloaders, defining the loss function and optimizer, and the main training loop with progress bars for training, validation, and testing.

Recurrent (predict.py)
This script will use the best module to recurrent the best result. It can also load data and visualize via tensorboard.

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
In the SiameseNetworkDataset class of dataset.py, some data augmentation code is commented out. If you wish to enhance the model's generalization capabilities, consider uncommenting this code.
Ensure the directory structure for the image data is correct when using the dataset class.
