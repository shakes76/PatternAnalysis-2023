import os
import random
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torch.nn.functional import pairwise_distance

from dataset import train_dataset, test_dataset, SiameseNetworkDataset, transform
from modules import SiameseResNet


if __name__ == '__main__':
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create data loaders for training and testing datasets
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=32)
    test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=16)

    # Initialize tensorboard writer
    writer = SummaryWriter()

    # Initialize the Siamese network and move it to the appropriate device
    net = SiameseResNet().to(device)

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()

    # Define the optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    # Set the network to training mode
    net.train()

    # Initialize the best loss to a very large number
    best_loss = float('inf')

    # Number of epochs to wait for improvement
    patience = 5

    # Number of epochs with no improvement after which training will be stopped
    no_improve_epochs = 0

    # Define source and validation directories
    src_dir = "AD_NC/train"
    val_dir = "AD_NC/val"

    # Check the number of images in the validation directory for each category
    num_images_in_val_AD = len(os.listdir(os.path.join(val_dir, "AD"))) if os.path.exists(os.path.join(val_dir, "AD")) else 0
    num_images_in_val_NC = len(os.listdir(os.path.join(val_dir, "NC"))) if os.path.exists(os.path.join(val_dir, "NC")) else 0

    # Only proceed with the transfer if there are less than 1000 images for each category in the validation set
    if num_images_in_val_AD < 1000 or num_images_in_val_NC < 1000:
        # Create the validation directory if it doesn't exist
        if not os.path.exists(val_dir):
            os.makedirs(val_dir)

        # Create the sub-folders for each category in the validation directory
        for category in ["AD", "NC"]:
            os.makedirs(os.path.join(val_dir, category), exist_ok=True)

            # Get the list of images in the source directory
            images = os.listdir(os.path.join(src_dir, category))

            # Select 1000 images at random
            num_images_to_transfer = 1000 - len(os.listdir(os.path.join(val_dir, category)))
            selected_images = random.sample(images, num_images_to_transfer)

            # Move the selected images to the validation directory
            for image in selected_images:
                src_path = os.path.join(src_dir, category, image)
                dst_path = os.path.join(val_dir, category, image)
                shutil.move(src_path, dst_path)

        # Print a message
        print("Validation set created!")
    else:
        print("Validation set already has enough images!")

    # Re-instantiate the train_dir and train_dataset after moving the images
    train_dir = ImageFolder(root=os.path.join(os.getcwd(), "AD_NC/train"))
    train_dataset = SiameseNetworkDataset(root_dir=os.path.join(os.getcwd(), "AD_NC/train"), transform=transform)

    # Create data loaders for training and testing datasets AFTER re-instantiating train_dataset
    train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=4, batch_size=32)
    test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=16)

    # Create the validation dataset
    val_dataset = SiameseNetworkDataset(root_dir=val_dir, transform=transform)

    # Create the validation data loader
    val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=4, batch_size=32)

    # Training loop with early stopping
    for epoch in range(0, 10):
        # Initialize the running loss to 0
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            # Get the images and labels from the data
            img0, img1, label = data

            # Move the images and labels to the appropriate device
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Compute predicted outputs by passing inputs to the model
            output1, output2 = net(img0, img1)

            # Compute the pairwise distance between output1 and output2
            distance = pairwise_distance(output1, output2)

            # Reshape the distance tensor to match the shape of the label tensor
            distance = distance.view(-1, 1)

            # Calculate the loss
            loss = criterion(distance, label)

            # Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Write the loss value to tensorboard
            running_loss += loss.item()

        # Calculate the average loss over the entire training data
        avg_loss = running_loss / len(train_dataloader)
        # Print the average loss value
        print("Epoch: {}, Loss: {}".format(epoch, avg_loss))
        # Write the average loss value to tensorboard
        writer.add_scalar('Training Loss', avg_loss, epoch)

        # Early stopping
        # If the current loss is less than the best loss, set the best loss to the current loss
        if avg_loss < best_loss:
            # Save the model parameters
            best_loss = avg_loss
            # Reset the number of epochs with no improvement
            no_improve_epochs = 0
        else:
            # Otherwise increment the number of epochs with no improvement
            no_improve_epochs += 1

        # If the number of epochs with no improvement has reached the patience limit, stop training
        if no_improve_epochs >= patience:
            print("Early stopping!")
            break

    # Save the trained model
    torch.save(net.state_dict(), "model.pth")
