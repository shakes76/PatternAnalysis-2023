import math
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import get_loader, CustomDataset
from modules import UNet

class Train:
    def __init__(self, model, train_loader, learning_rate, num_epochs, save_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.save_path = save_path

        # Define loss and optimizer
        self.dice_loss = DiceLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            total_dice = 0.0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Calculate the iteration number within the epoch
                iteration = batch_idx

                # Forward pass
                outputs = self.model(inputs)

                # Check for values greater than 1 in both predicted and target tensors
                if (outputs > 1).any() or (targets > 1).any():
                    print("Values greater than 1 found in outputs or targets.")
                    # Handle or log the issue and continue

                # Calculate the Dice coefficient
                dice = self.calculate_dice(outputs, targets)
                total_dice += dice

                if torch.isnan(dice).any():
                    print("Dice coefficient contains NaN values.")
                    # Handle or log the issue and continue

                # Backpropagation
                self.optimizer.zero_grad()
                dice_loss = dice
                dice_loss.backward()
                self.optimizer.step()

                # Print the training progress
                #print(f"Epoch {epoch + 1}/{self.num_epochs}, Iteration {iteration}, Loss: {dice.item()}")

            average_dice = total_dice / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Average Loss: {average_dice}")

        # Save the complete model after training
        self.save_model()

    def calculate_dice(self, predicted, target, smooth=1e-5):
        iflat = predicted.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))
    def save_model(self):
        save_path = f"{self.save_path}.pt"
        torch.save(self.model, save_path)
        print(f"Model saved as {save_path}")

class DiceLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, predicted, target):
        smooth = 1.
        iflat = predicted.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


if __name__ == '__main__':
    train_input_dir = r"C:\Users\sam\Downloads\ISIC2018_Task1-2_SegmentationData_x2\ISIC2018_Task1-2_Training_Input_x2"
    train_mask_dir = r"C:\Users\sam\Downloads\ISIC2018_Task1-2_SegmentationData_x2\ISIC2018_Task1_Training_GroundTruth_x2"
    batch_size = 4
    num_workers = 4
    learning_rate = 0.9
    num_epochs = 10
    save_path = r"C:\Users\sam\OneDrive\Desktop\COMP\Lab-Report"  # Set the desired save path

    # Create the data loader using your CustomDataset and get_loader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_loader = get_loader(train_input_dir, train_mask_dir, batch_size, num_workers, transform=transform)

    model = UNet(3, 1)

    trainer = Train(model, train_loader, learning_rate, num_epochs, save_path)
    trainer.train()

