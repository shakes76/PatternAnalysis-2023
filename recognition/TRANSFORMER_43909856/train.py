import os
import os.path as osp
import torch
import torch.nn as nn
import time
from timm.data import ImageDataset
from timm.data.transforms_factory import create_transform

import dataset
import modules


"""
This file contains code for training, validating, testing and saving the model. 
The ViT model is imported from modules.py, and the data loader
will be imported from dataset.py. The losses and metrics will be plotted during
training.

This file currently tests the model on a benchmark dataset (Imagenette) to confirm
that model components work correctly.
"""


#### Set-up GPU device ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning: CUDA not found. Using CPU")
else:
    print(torch.cuda.get_device_name(0))


#### Model hyperparameters: ####
N_EPOCHS = 80
LEARNING_RATE = 0.001
N_CLASSES = 10
BATCH_SIZE = 32


#### File paths: ####
dataset_path = "./recognition/TRANSFORMER_43909856/imagenette"
output_path = "./recognition/TRANSFORMER_43909856/models"

"""
 https://huggingface.co/blog/fine-tune-vit
 https://openaccess.thecvf.com/content/CVPR2023W/ECV/papers/Bhattacharyya_DeCAtt_Efficient_Vision_Transformers_With_Decorrelated_Attention_Heads_CVPRW_2023_paper.pdf
 https://ieeexplore.ieee.org/document/9880094
"""

# Get the training data (Imagenette)
tfm = create_transform(224, is_training=True)
imagenette_ds = ImageDataset(osp.join(dataset_path, "imagenette2-320", "train"), transform=tfm)
train_loader = torch.utils.data.DataLoader(imagenette_ds, batch_size=BATCH_SIZE, shuffle=True)

# Get the testing data (Imagenette)
tfm_val = create_transform(224, is_training=False)
imagenette_val_ds = ImageDataset(osp.join(dataset_path, "imagenette2-320", "val"), transform=tfm_val)
test_loader = torch.utils.data.DataLoader(imagenette_val_ds, batch_size=BATCH_SIZE, shuffle=False)


# Initalise the model
model = modules.SimpleViT(image_size=(224, 224), patch_size=(16, 16), n_classes=10, dimensions=384,
                            depth=12, n_heads=6, mlp_dimensions=1536)
# Move the model to the GPU device
model = model.to(device)


criterion = nn.CrossEntropyLoss()

# Use the Adam optimiser for ViT
# optimiser = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
# Use a piecewise linear LR scheduler
total_step = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=LEARNING_RATE, 
                                                steps_per_epoch=total_step, epochs=N_EPOCHS)
# ViT paper uses a different kind of LR scheduler - may want to try this


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
        
        # Print the training metrics for every 5 images
        if (i+1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{N_EPOCHS}] Step [{i+1}/{total_step}] " +
                  f"Training loss: {round(loss.item(), 5)}")
            
        # Step through the learning rate scheduler
        scheduler.step()

# Get the amount of time that the model spent training
end_time = time.time()
elapsed_time = end_time - start_time

print(f"Training finished. Training took {elapsed_time} seconds ({elapsed_time/60} minutes)")


# Create a dir for saving the trained model (if one doens't exist)
if not osp.isdir(output_path):
    os.makedirs(output_path)

# Save the model
torch.save(model.state_dict(), osp.join(output_path, "ViT_imagenette_model.pt"))

