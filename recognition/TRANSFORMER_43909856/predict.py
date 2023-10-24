import os
import os.path as osp
import torch
import torch.nn as nn
import time

import dataset
import modules

"""
This file is used to test the ViT model trained on the ADNI dataset.
Any results will be printed out, and visualisations will be provided 
where applicable.
"""
# TODO add plots of metrics - could do confusion matrix, ROC or Precision/Recall
# curve.

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
DATASET_PATH = osp.join("recognition", "TRANSFORMER_43909856", "dataset", "AD_NC")
OUTPUT_PATH = osp.join("recognition", "TRANSFORMER_43909856", "models")


"""
Loads the ADNI dataset's test set.
Loads the previously trained ViT classification model, then tests the model
on the test set.
"""
def test_model():
    # Get the testing data (ADNI)
    test_loader = dataset.load_ADNI_data()

    # Initalise a blank slate model
    model = modules.SimpleViT(image_size=(IMG_SIZE, IMG_SIZE), patch_size=(16, 16), n_classes=N_CLASSES, 
                            dimensions=384, depth=12, n_heads=6, mlp_dimensions=1536, n_channels=3)
    # Move the model to the GPU device
    model = model.to(device)
    # Load the pre-trained model into the blank slate ViT
    model.load_state_dict(torch.load(osp.join(OUTPUT_PATH, "ViT_ADNI_model.pt"), map_location=device))


    # Test the model:
    print("Testing has started")
    # Get a timestamp for when the model testing starts
    start_time = time.time()

    model.eval()
    with torch.no_grad():
        # Keep track of the total number predictions vs. correct predictions
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Get the amount of time that the model spent testing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Test accuracy: {(100 * correct) / total} %")
    print(f"Testing finished. Testing took {elapsed_time} seconds ({elapsed_time/60} minutes)")


"""
Main method - make sure to run any methods in this file within here.
Adding this so that multiprocessing runs appropriately/correctly
on Windows devices.
"""
def main():
    test_model()

if __name__ == '__main__':    
    main()