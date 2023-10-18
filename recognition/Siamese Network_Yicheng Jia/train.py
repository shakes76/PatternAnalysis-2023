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
from tqdm import tqdm

from dataset import test_dataset, SiameseNetworkDataset, transform
from modules import SiameseResNet, SiameseVGG


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = nn.functional.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


if __name__ == '__main__':
    # Check if CUDA is available and set the device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    # Initialize tensorboard writer
    writer = SummaryWriter()

    # Initialize the Siamese network and move it to the appropriate device
    net = SiameseResNet().to(device)
    #net = SiameseVGG().to(device)

    # Define the loss function
    criterion = ContrastiveLoss()

    # Define the optimizer using SGD
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # Define the learning rate scheduler, using StepLR
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Set the network to training mode
    net.train()

    # Initialize the best loss to a very large number
    best_val_loss = float('inf')

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
    test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=4, batch_size=32)

    # Create the validation dataset
    val_dataset = SiameseNetworkDataset(root_dir=val_dir, transform=transform)

    # Create the validation data loader
    val_dataloader = DataLoader(val_dataset, shuffle=True, num_workers=4, batch_size=32)

    # Training loop with early stopping
    for epoch in range(0, 20):
        # Initialize the running loss to 0
        running_loss = 0.0
        correct = 0
        total = 0

        # Progress bar
        for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            # Get the images and labels from the data
            img0, img1, label = data

            # Move the images and labels to the appropriate device
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Compute predicted outputs by passing inputs to the model
            output1, output2 = net(img0, img1)

            # Calculate the loss
            loss = criterion(output1, output2, label)

            # Backward pass: Compute gradient of the loss with respect to model parameters
            loss.backward()

            # Update the model parameters
            optimizer.step()

            # Write the loss value to tensorboard
            running_loss += loss.item()

            # Calculate the Euclidean distance between the two outputs
            distance = pairwise_distance(output1, output2).view(-1, 1)

            # Calculate the accuracy
            predicted = (torch.sigmoid(distance) > 0.5).float()
            total += label.size(0)
            correct += (predicted == label).sum().item()

        # Calculate the average loss over the entire training data
        avg_loss = running_loss / len(train_dataloader)
        accuracy = 100 * correct / total
        print(f"Epoch: {epoch}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Write the average loss value to tensorboard
        writer.add_scalar('Training Loss', avg_loss, epoch)
        writer.add_scalar('Training Accuracy', accuracy, epoch)

        # Validation loop
        val_loss = 0.0
        for i, data in enumerate(val_dataloader):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            with torch.no_grad():
                output1, output2 = net(img0, img1)
                loss = criterion(output1, output2, label)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Validation - Epoch: {epoch}, Loss: {avg_val_loss:.4f}")

        # Early stopping
        # If the current loss is less than the best loss, set the best loss to the current loss
        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print("Early stopping!")
            break

        # Step the scheduler
        scheduler.step(avg_val_loss)

    # Save the trained model
    torch.save(net.state_dict(), "model.pth")
