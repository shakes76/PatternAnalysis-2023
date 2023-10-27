import os
import os.path as osp
import torch
import torch.nn as nn
import time
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt

import dataset
import modules

"""
This file is used to test the ViT model trained on the ADNI dataset.
Any results will be printed out, and visualisations will be provided 
where applicable.
"""

#### Set-up GPU device ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")
else:
    print(torch.cuda.get_device_name(0))


#### Model hyperparameters: ####
BATCH_SIZE = 32
N_CLASSES = 2
# Dimensions to resize the original 256x240 images to (IMG_SIZE x IMG_SIZE)
IMG_SIZE = 224


#### File paths: ####
# Local dataset path
# DATASET_PATH = osp.join("recognition", "TRANSFORMER_43909856", "dataset", "AD_NC")
# Path to dataset on Rangpur HPC
DATASET_PATH = osp.join("/", "home", "groups", "comp3710", "ADNI", "AD_NC")
OUTPUT_PATH = osp.join("recognition", "TRANSFORMER_43909856", "models")


"""
Loads the ADNI dataset's test set.
Loads the previously trained ViT classification model, then tests the model
on the test set.

Params:
    model_filename (str): The file path and file name for the model to be evaluated
    save_metrics (bool): If true, saves separate lists of the model's
                         predicted values and the corresponding observed/empirical
                         values for each image in the test set (to CSV files). 
                         Otherwise, does not save these values
"""
def test_model(model_filename=osp.join(OUTPUT_PATH, "ViT_ADNI_model.pt"), save_metrics=True):
    # Get the testing data (ADNI)
    test_data = dataset.load_ADNI_data(dataset_path=DATASET_PATH)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Initalise a blank slate model
    model = modules.SimpleViT(image_size=(IMG_SIZE, IMG_SIZE), patch_size=(16, 16), n_classes=N_CLASSES, 
                            dimensions=384, depth=12, n_heads=6, mlp_dimensions=1536, n_channels=3)
    # Move the model to the GPU device
    model = model.to(device)
    # Load the pre-trained model into the blank slate ViT
    model.load_state_dict(torch.load(model_filename, map_location=device))


    # Test the model:
    print("Testing has started")
    # Get a timestamp for when the model testing starts
    start_time = time.time()

    # Store the model's predicted classes and the observed/empirical classes
    predictions = []
    observed = []
    
    model.eval()
    with torch.no_grad():
        # Keep track of the total number predictions vs. correct predictions
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            # Add images to the data and get the predicted classes
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            # Add to the total # of predictions
            total += labels.size(0)
            # Add correct predictions to a total
            correct += (predicted == labels).sum().item()

            # Save the predictions and the observed/empirical class labels
            predictions += predicted.cpu()
            observed += labels.cpu()

    # Get the amount of time that the model spent testing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Test accuracy: {round((100 * correct) / total, 5)}%")
    print(f"Testing finished. Testing took {round(elapsed_time, 2)} seconds "
          +f"({round(elapsed_time/60, 4)} minutes)")

    # Save testing metrics
    if save_metrics:
        # Create a dir for saving the testing metrics (if one doesn't exist)
        if not osp.isdir(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        # Save the model's predictions
        np.savetxt(osp.join(OUTPUT_PATH, 'ADNI_test_predictions.csv'), 
                   np.asarray(predictions))
        # Save the observed/empirical values for the test set
        np.savetxt(osp.join(OUTPUT_PATH, 'ADNI_test_observed.csv'), 
                   np.asarray(observed))


"""
Load the predicted classes or the empirical/observed classes, for each image in 
the test set. These values are loaded from CSV files, which are saved during the
testing process.

Params:
    filename (str): the name of the CSV file to load
Returns:
    An array, in which each entry is a classification label (either predicted
    or actual), as predicted by the dataset. Labels are either a 0 (AD) or 1 (NC)
"""
def load_test_labels(filename=osp.join(OUTPUT_PATH, 'ADNI_test_predictions.csv')):
    # Load the file
    labels = np.loadtxt(filename, dtype=float)
    # Convert from a numpy array to a python base lib list
    return labels.tolist()


"""
Plot a confusion matrix, based on the predicted classes and empirical/observed
classes for the test set data.

Params:
    predicted (array[int]): predicted class values for each image in the test set.
                            Values are either 0 (AD) or 1 (NC)
    observed (array[int]): empirical/observed class values for each image in the 
                            test set. Values are either 0 (AD) or 1 (NC)
    show_plot (bool): show the plot in a popup window if True; otherwise, don't
                      show the plot
    save_plot (bool): save the plot as a PNG file to the directory "plots" if
                      True; otherwise, don't save the plot
"""
def plot_confusion_matrix(predicted, observed, show_plot=False, save_plot=False):
    cm = confusion_matrix(observed, predicted)
    # Create a graph/plot of the confusion matrix
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["AD", "NC"])
    cm_display.plot()
    # Add a title
    plt.title("ViT Transformer (ADNI classifier) test set confusion matrix")

    # Save the plot
    if save_plot:
        # Create an output folder for the plot, if one doesn't already exist
        directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'plots')
        if not os.path.exists(directory):
            os.makedirs(directory) 
        # Save the plot in the "plots" directory
        plt.savefig(os.path.join(directory, "ViT_test_confusion_matrix.png"), dpi=600)
    if show_plot:
        # Show the plot
        plt.show()


"""
Main method - make sure to run any methods in this file within here.
Adding this so that multiprocessing runs appropriately/correctly
on Windows devices.
"""
def main():
    # Test the model
    # test_model()

    # Load predicted class labels
    predicted = load_test_labels()
    # Load empirical/observed class labels
    observed = load_test_labels(osp.join(OUTPUT_PATH, 'ADNI_test_observed.csv'))

    # Plot a confusion matrix
    plot_confusion_matrix(predicted, observed, show_plot=True, save_plot=True)


if __name__ == '__main__':
    main()