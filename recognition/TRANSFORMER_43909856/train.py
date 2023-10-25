import os
import os.path as osp
import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.backward_compatibility import worker_init_fn

import dataset
import modules


"""
This file contains code for training, validating, testing and saving the model. 
The ViT model is imported from modules.py, and the data loader
is imported from dataset.py. 
The losses and metrics will be plotted during training.


 https://huggingface.co/blog/fine-tune-vit
 https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Bhattacharyya_DeCAtt_Efficient_Vision_Transformers_With_Decorrelated_Attention_Heads_CVPRW_2023_paper.pdf
 https://ieeexplore.ieee.org/document/9880094

"""

#### Set-up GPU device ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")
else:
    print(torch.cuda.get_device_name(0))


#### Model hyperparameters: ####
N_EPOCHS = 2
LEARNING_RATE = 0.001
N_CLASSES = 2
# Dimensions to resize the original 256x240 images to (IMG_SIZE x IMG_SIZE)
IMG_SIZE = 224
# The batch size used by the data loaders for the train, validation, and test sets
BATCH_SIZE = 32


#### File paths: ####
DATASET_PATH = osp.join("recognition", "TRANSFORMER_43909856", "dataset", "AD_NC")
OUTPUT_PATH = osp.join("recognition", "TRANSFORMER_43909856", "models")


"""
Loads the ADNI dataset into train (and possibly validation) sets.
Initialises the model, then trains the model.

Params:
    save_model_data (bool): if true, saves the model as a .pt file and model 
                            training metrics as .csv files. If false, doesn't
                            save the model or training metrics
"""
def train_model(save_model_data=True):
    # Get the training and validation data (ADNI) and # of total steps
    train_images, total_step, val_images = \
                dataset.load_ADNI_data_per_patient(dataset_path=DATASET_PATH, train_size=0.8)

    # Add the training and (if any) validation data to data loaders
    train_loader = DataLoader(train_images.dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, worker_init_fn=worker_init_fn)
    if val_images is not None:
        # If val_images is None, don't create a validation set
        val_loader = DataLoader(val_images.dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=2, worker_init_fn=worker_init_fn)


    # Need to add some transforms to the input data:
    # TODO look into converting images from RGB to Greyscale, as individual channels' data seems to be irrelevant for black and white MRI images

    # Initalise the model
    model = modules.SimpleViT(image_size=(IMG_SIZE, IMG_SIZE), patch_size=(16, 16), n_classes=N_CLASSES, 
                            dimensions=384, depth=12, n_heads=6, mlp_dimensions=1536, n_channels=3)
    # Move the model to the GPU device
    model = model.to(device)

    # Use binary cross-entropy as the loss function
    criterion = nn.CrossEntropyLoss()

    # Use the Adam optimiser for ViT
    # optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Use a piecewise linear LR scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=LEARNING_RATE, 
                                                    steps_per_epoch=total_step, epochs=N_EPOCHS)
    # TODO ViT paper uses a different kind of LR scheduler - may want to try this
    # TODO do hyperparameter tuning on number of epochs
    # Use validation accuracy to determine the ideal model
    # Also output validation loss for comparisons (see if model is overfitting/underfitting)

    # Store the epoch, step, & loss value for the model at this epoch & step
    train_loss_values = []

    # Store the model's predicted classes and the observed/empirical classes
    predictions = []
    observed = []

    # Train the model:
    model.train()
    print("Training has started")
    # Get a timestamp for when the model training starts
    start_time = time.time()

    # Train the model for the given number of epochs
    for epoch in range(N_EPOCHS):

        # Train on each image in the training set
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Perform a forward pass of the model
            outputs = model(images)
            # Calculate the training loss
            loss = criterion(outputs, labels)

            # Perform backpropagation
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            
            # Print/log the training metrics for every 20 steps, and at the end of each epoch
            if (i+1) % 20 == 0 or i+1 == total_step:
                print(f"Epoch [{epoch+1}/{N_EPOCHS}] Step [{i+1}/{total_step}] " +
                    f"Training loss: {round(loss.item(), 5)}")
                train_loss_values += [[epoch+1, i+1, total_step, round(loss.item(), 5)]]
            
            # Get predictions on the training set data from the model
            _, predicted = torch.max(outputs.data, 1)
            # Save the predictions and the observed/empirical class labels
            predictions += predicted
            observed += labels

            # Step through the learning rate scheduler
            scheduler.step()

    # Get the amount of time that the model spent training
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Training finished. Training took {elapsed_time} seconds ({elapsed_time/60} minutes)")

    if save_model_data:
        # Create a dir for saving the trained model (if one doesn't exist)
        if not osp.isdir(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        # Save the model
        torch.save(model.state_dict(), osp.join(OUTPUT_PATH, "ViT_ADNI_model.pt"))

        # Save the training loss values
        np.savetxt(osp.join(OUTPUT_PATH, 'ADNI_train_loss.csv'), 
                   np.asarray(train_loss_values))
        
        # Save the model's predictions
        np.savetxt(osp.join(OUTPUT_PATH, 'ADNI_train_predictions.csv'), 
                   np.asarray(predictions))
        # Save the observed/empirical values for the training set
        np.savetxt(osp.join(OUTPUT_PATH, 'ADNI_train_observed.csv'), 
                   np.asarray(observed))


"""
Main method - make sure to run any methods in this file within here.
Adding this so that multiprocessing runs appropriately/correctly
on Windows devices.
"""
def main():
    train_model()

if __name__ == '__main__':    
    main()




