"""
File: train.py
Author: Arshia Sharma 
Description: Trains ImprovedUnet architecture. 

Dependencies: torch

"""
# import libraries. 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.transforms.functional as tnsf
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# import local files. 
from dataset import ISICDataset
from modules import ImprovedUnet

# UPDATE WITH YOUR FILE PATHS TO DATA.
TRAIN_DATA_PATH  = "ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
TRAIN_MASK_PATH = "ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"

# GLOBAL CONSTANTS
BATCH_SIZE = 2
LEARNING_RATE = 0.0005

"""
Trains Improved Unet and outputs training and validation losses. 

Parameters:
- model (nn.Module): Improved unet architecture 
- train_loader (DataLoader): DataLoader for the training dataset.
- valid_loader (DataLoader): DataLoader for the validation dataset.
- num_epochs (int, optional): Number of training epochs (default is 100).
- device (str, optional): Device to perform training (default is "cuda" for GPU).

Returns:
- model (nn.Module): The trained deep learning model.
- training_losses (list): List of training losses over epochs.
- validation_losses (list): List of validation losses over epochs.
"""
def train(model, train_loader, valid_loader, num_epochs=100, device="cuda"):
    # set up criterion, optimiser, and scheduler for learning rate. 
    criterion = dice_coefficient
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.985)

    model.to(device)
    model.train()

    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            inputs, masks = inputs.to(device), masks.to(device)
            optimiser.zero_grad()
            outputs = model(inputs)

            # we want to maximise the dice coefficient
            # loss is then 1 - dice coefficient 
            loss = 1 - criterion(outputs, masks) 
            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_loss = 0.0

        # get validation losses. 
        with torch.no_grad():
            for val_inputs, val_masks in valid_loader:
                val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)
                val_outputs = model(val_inputs)
                val_loss += 1 - criterion(val_outputs, val_masks).item()

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(valid_loader)}")

        training_losses.append(running_loss / len(train_loader))
        validation_losses.append(val_loss / len(valid_loader))

    return model, training_losses, validation_losses

"""
    Calculate the Dice Coefficient between predicted and ground truth mask. 

    The Dice Coefficient measures the similarity or overlap between two binary sets (in our case, images).

    Parameters:
    - y_true (tensor): Ground truth binary tensor.
    - y_pred (tensor): Predicted binary tensor.
    - eps (float, optional): A small constant to prevent division by zero (default is 10^-8).

    Returns:
    - dice (float): The Dice Coefficient score, ranging from 0 to 1, with 1 indicating perfect overlap.
"""
def dice_coefficient(y_true, y_pred, eps=10**-8):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + eps) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + 1.)
    return dice

""""
Plots training and validation losses on the same plot. 
Outputs plot. 

"""
def plot_losses(train_losses, valid_losses):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Losses over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


"""
Helper function not directly used for training. 
Sanity checker that displays image and its corresponding ground truth mask.

"""
def show_sample(image, mask):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Convert image and mask tensors to PIL images
    image = tnsf.to_pil_image(image)
    mask = tnsf.to_pil_image(mask)

    axes[0].imshow(image)
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Mask')
    axes[1].axis('off')

    plt.show()

"""
    Loads in data and creates training and validation dataloaders 
    
    Parameters:
    - img_path (str): Path to the directory containing image files.
    - labels_path (str): Path to the directory containing labels or annotations.
    - transform (callable): Data transformation to be applied to the images.
    - batch_size (int): Batch size for data loaders.

    Returns:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
"""
def load_data(img_path, labels_path, transform, batch_size):
    seed = 1243535  # You can use any integer value as your seed
    torch.manual_seed(seed)
    #@random.seed(seed)


    dataset = ISICDataset(img_path, labels_path, transform)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

if __name__ == "__main__":
    # connect to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")
    
    # create model. 
    model = ImprovedUnet()

    # set up data transform. 
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.7071, 0.5821, 0.5360], std=[0.1561, 0.1644, 0.1795])
    ])

    # load data
    train_loader, valid_loader = load_data(TRAIN_DATA_PATH, TRAIN_MASK_PATH, data_transform, batch_size=1)

    # optional - view sample input and ground truth. 
    #sample_batch = next(iter(train_loader))
    #sample_image, sample_mask = sample_batch[0][0], sample_batch[1][0]  
    #show_sample(sample_image, sample_mask)

    # train improved unet
    trained_model, training_losses, validation_losses = train(model, train_loader, valid_loader, 
                                                              device=device, num_epochs=50)

    # Save the trained model for predictions
    torch.save(trained_model, "improved_UNET.pth")

    # plot train and validation loss
    plot_losses(training_losses, validation_losses)