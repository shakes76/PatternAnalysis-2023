# Siamese Network for Alzheimer's Disease Classification

This project uses a Siamese Network to classify Alzheimer's disease based on image pairs.

## Files Description:

1. **modules.py**: 
    - Contains the Siamese Network architecture.
    - Uses a pre-trained ResNet50 as the backbone.
    - Defines the forward pass for single images and image pairs.

2. **dataset.py**: 
    - Defines the Siamese Network dataset.
    - Loads image pairs and labels.
    - Includes data augmentation and preprocessing steps.

3. **train.py**: 
    - Contains the training loop for the Siamese Network.
    - Implements early stopping based on validation loss.
    - Logs training loss to TensorBoard.

4. **predict.py**: 
    - Uses the trained model to make predictions on the test set.
    - Calculates and prints the accuracy of the model.
    - Logs test accuracy to TensorBoard.

## How to Run:

1. Ensure you have all the required libraries installed.
2. Make sure the path of the dataset we need has been set correctly, the folder will be named "AD_NC" and be placed together with all of the py files. 
3. Run the scripts in the following order:
    - `train.py`: This will train the model and save the best weights.
    - `predict.py`: This will use the trained model to make predictions on the test set.
4. After running the above script, your AD_NC directory structure should be as follows:
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


