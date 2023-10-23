import os
import os.path as osp
import torch
import torch.nn as nn
import time

import dataset

"""
This file is used to test the ViT model trained on the ADNI dataset.
Any results will be printed out, and visualisations will be provided 
where applicable.
"""
# TODO add plots of loss/metrics

#### Set-up GPU device ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")
else:
    print(torch.cuda.get_device_name(0))


#### Model hyperparameters: ####
BATCH_SIZE = 32


#### File paths: ####
dataset_path = "./recognition/TRANSFORMER_43909856/dataset"
output_path = "./recognition/TRANSFORMER_43909856/models"


# Get the testing data (ADNI)
test_loader = dataset.load_ADNI_data()

# Load the model
model = torch.load(osp.join(output_path, "ViT_ADNI_model.pt"))
# Move the model to the GPU device
model = model.to(device)


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
        correct += (predicted == labels).sum().items()

# Get the amount of time that the model spent testing
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Test accuracy: {(100 * correct) / total} %")
print(f"Training finished. Training took {elapsed_time} seconds ({elapsed_time/60} minutes)")

