import dataset
import modules
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import statistics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
import os

def dice_loss(output, target):
    """
    Compute the Dice loss between the predicted output and target.
    
    Parameters:
        output (Tensor): The model's prediction for the input image(s),
        target (Tensor): The ground truth segmentation masks.
        
    Returns:
        float: The computed Dice loss.
    """
    epsilon = 10**-8 
    intersection = (output * target).sum()
    denominator = (output + target).sum() + epsilon
    loss = 1 - (2.0 * intersection) / (denominator)
    return loss

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper Parameters
num_epochs = 30
initial_lr = 5e-4
batch_size = 16
lr_decay = 0.985
l2_weight_decay = 1e-5

# define model
model = modules.UNet2D(3, 1).to(device)

# Datasets & Loaders
train_dataset = dataset.ISICDataset(dataset_type='training', transform=dataset.get_transform())
val_dataset = dataset.ISICDataset(dataset_type='validation', transform=dataset.get_transform())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Optimisation & Loss Settings
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=l2_weight_decay)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay)

# Starting the training time measurement
start_time = time.time()

# Lists to store the loss and dice coefficient values for both training and validation phases
train_losses = []
train_dice = []
validation_losses = []
validation_dice = []

# Initializing current epoch counter
curr_epoch = 0

# Iterating through the epochs
for epoch in range(num_epochs):
    # Incrementing the current epoch counter
    curr_epoch += 1
    
    # Lists to store the loss and dice coefficients for each batch
    losses = []
    coeffs = []
    running_loss = 0.0

    # --- TRAINING PHASE ---
    # During the training phase, the model learns by adjusting its 
    # weights to minimize the loss on the training data. The training data is 
    # passed through the model (forward pass) to compute the output and calculate the loss. 
    # The gradients of the loss with respect to model parameters are computed (backward pass), and the 
    # optimizer updates the model's weights. Training loss is monitored, and if applicable, 
    # the model's performance on some validation data is checked periodically to prevent overfitting. 

    model.train()

    # Iterating through each batch of training data
    for i, data in enumerate(train_loader, 0):
        print("train: ", i)
        # Moving the images and labels to the device (e.g., GPU)
        images, labels = data['image'].to(device), data['mask'].to(device)

        # Forward pass
        outputs = model(images)
        # Calculating the dice loss
        loss = dice_loss(outputs, labels)
        
        # Appending loss and coefficient to lists
        losses.append(loss.item())
        coeffs.append((1 - dice_loss(outputs, labels)).item())

        # Zeroing the parameter gradients
        optimizer.zero_grad()
        # Backward pass
        loss.backward()
        # Updating the parameters
        optimizer.step()
        running_loss += loss.item()

        # Logging the running loss for every 10 batches
        if i % 10 == 9:
            print(f"loss: {running_loss / 10:.3f}")
            running_loss = 0.0
        
    # Adjusting the learning rate based on the scheduler
    scheduler.step()
    
    # Calculating the mean loss and coefficient for the epoch
    train_loss = statistics.mean(losses)
    train_coeff = statistics.mean(coeffs)

    # --- VALIDATION PHASE ---
    # The validation phase aims to evaluate the model's performance 
    # on a separate dataset that was not used during training, which 
    # helps in assessing how well the model is generalizing to unseen data. 
    # During validation, the model is set to evaluation mode, and gradient 
    # computation is typically disabled to speed up computations and use less memory. 
    # The model makes predictions on the validation data, and the validation loss is 
    # computed and stored for later analysis.

    losses = []
    coeffs = []

    # Setting the model to evaluation mode
    model.eval()
    # Disabling gradient computation during validation to save memory
    with torch.no_grad():
        # Iterating through each batch of validation data
        for i, data in enumerate(val_loader, 0):
            print("validate ", i)
            # Moving the images and labels to the device
            images, labels = data['image'].to(device), data['mask'].to(device)

            # Forward pass
            outputs = model(images)
            # Calculating the dice loss
            loss = dice_loss(outputs, labels)
            # Appending loss and coefficient to lists
            losses.append(loss.item())
            coeffs.append((1 - dice_loss(outputs, labels)).item())

            # Saving the model and plots every 5 epochs or the first epoch
            if (curr_epoch % 5 == 0 or curr_epoch == 1):
                # Defining the model checkpoint path and saving the model
                save_path = os.path.join("model_checkpoints", f"model_epoch_{epoch+1}.pth")
                if not os.path.exists("model_checkpoints"):
                    os.makedirs("model_checkpoints")
                torch.save(model.state_dict(), save_path)

                # Preparing the directory for plot saving
                if not os.path.exists('plots'):
                    os.makedirs('plots')

                # Creating subplots for displaying original images, actual masks, 
                # and predicted masks for some samples in the batch
                fig, axs = plt.subplots(5, 3, figsize=(15,5*5))
                axs[0][0].set_title("Original Image")
                axs[0][1].set_title("Actual Mask")
                axs[0][2].set_title("Predicted Mask")

                # Plotting and saving the image-mask-prediction triplets
                for row in range(5):
                    img = images.cpu()[row].permute(1,2,0).numpy()
                    label = labels.cpu()[row].permute(1,2,0).numpy()
                    pred = outputs.cpu()[row].permute(1,2,0).numpy()

                    axs[row][0].imshow(img)
                    axs[row][0].xaxis.set_visible(False)
                    axs[row][0].yaxis.set_visible(False)

                    axs[row][1].imshow(label, cmap="gray")
                    axs[row][1].xaxis.set_visible(False)
                    axs[row][1].yaxis.set_visible(False)

                    axs[row][2].imshow(pred, cmap="gray")
                    axs[row][2].xaxis.set_visible(False)
                    axs[row][2].yaxis.set_visible(False)
                
                fig.suptitle("Validation Segments Epoch: " + str(curr_epoch))
                plt.savefig(os.path.join('plots', "ValidationSegmentsEpoch" + str(curr_epoch)))
                plt.close()
    
    # Calculating the mean loss and coefficient for the validation
    valid_loss = statistics.mean(losses)
    valid_coeff = statistics.mean(coeffs)

    # Appending loss and coefficient values for logging and plotting
    train_losses.append(train_loss)
    train_dice.append(train_coeff)
    validation_losses.append(valid_loss)
    validation_dice.append(valid_coeff)

    # Logging the training and validation metrics for the epoch
    print ("epoch_num = {}/{}, Training Loss: {:.5f}, Training Dice Similarity {:.5f}".format(epoch+1, num_epochs, train_losses[-1], train_dice[-1]))
    print('Val Loss: {:.5f}, Val Average Dice Similarity: {:.5f}'.format(statistics.mean(validation_losses), statistics.mean(validation_dice)))

# Printing the total training time
print("Total Time: " + str((time.time() - start_time)/60) + " Minutes")
    
x = list(range(1, len(train_losses) + 1))

# Plotting the training and validation loss over epochs
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.plot(x, train_losses, label=f"Training Loss")
plt.plot(x, validation_losses, label=f"Validation Loss")
plt.legend()
plt.title(f"Training and Validation Loss Over Epochs")
plt.savefig(os.path.join('plots', "results_loss"))
plt.close()

# Plotting the training and validation dice coefficient over epochs
plt.xticks(np.arange(min(x), max(x)+1, 1.0))
plt.plot(x, train_dice, label=f"Training Dice Coefficient")
plt.plot(x, validation_dice, label=f"Validation Dice Coefficient")
plt.legend()
plt.title(f"Training and Validation Dice Coefficients Over Epochs")
plt.savefig(os.path.join('plots', "results_dice"))
plt.close()
