import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset import ISICDataset
from modules import ImprovedUNet
from torchvision import transforms
import matplotlib.pyplot as plt
import time


def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([transforms.ToTensor()])

    # Create the entire dataset
    full_dataset = ISICDataset("ISIC2018_Task1-2_Training_Input_x2", "ISIC2018_Task1_Training_GroundTruth_x2", image_transform, mask_transform)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, validation_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False)

    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1

    train_losses = []
    validation_losses = []
    validation_dscs = []

    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        total_dice_coefficient = 0.0

        # Training phase
        for step, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            dice_coefficient = 1 - loss.item()
            total_dice_coefficient += dice_coefficient
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        train_losses.append(epoch_loss)
        average_dice_coefficient = total_dice_coefficient / len(
            train_loader) 
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")
        print(f"Epoch {epoch + 1} Average Dice Coefficient: {average_dice_coefficient:.4f}\n")


        # Validation phase
        model.eval()
        val_loss = 0.0
        total_val_dice = 0.0
        with torch.no_grad():
            for images, masks in validation_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = dice_loss(outputs, masks)
                val_loss += loss.item() * images.size(0)
                dice_coefficient = 1 - loss.item()
                total_val_dice += dice_coefficient * images.size(0)

        avg_val_loss = val_loss / (len(validation_loader) * validation_loader.batch_size)
        avg_val_dice = total_val_dice / (len(validation_loader) * validation_loader.batch_size)

        validation_losses.append(avg_val_loss)
        validation_dscs.append(avg_val_dice)

        print(f"Epoch {epoch + 1} Validation Loss: {avg_val_loss:.4f}, Validation DSC: {avg_val_dice:.4f}")

    end_time = time.time()

    # Plotting
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, '-o', label='Training Loss')
    plt.plot(epochs, validation_losses, '-o', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, validation_dscs, '-o')
    plt.title('Validation DSC')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')

    plt.tight_layout()
    plt.show()

    print(f"Total Training Time: {end_time - start_time:.2f} seconds")
    torch.save(model.state_dict(), "model_checkpoint.pth")


if __name__ == "__main__":
    train()
