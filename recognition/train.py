import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import ISICDataset
from modules import ImprovedUNet
from torchvision import transforms


def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)

    # Flatten label and prediction tensors
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

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = ISICDataset("ISIC2018_Task1-2_Training_Input_x2", "ISIC2018_Task1_Training_GroundTruth_x2",
                                image_transform, mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1
    num_epochs = 1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        total_dice_coefficient = 0.0  # Initialize the accumulator for Dice coefficient

        # Print epoch number at the start of each epoch
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        for step, (images, masks) in enumerate(train_loader, 1):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = dice_loss(outputs, masks)

            dice_coefficient = 1 - loss.item()
            total_dice_coefficient += dice_coefficient  # Accumulate the Dice coefficient for this batch

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)  # multiply by batch size
            total_samples += images.size(0)

            # Print the loss for every step/batch
            print(f"Epoch {epoch + 1}/{num_epochs}, Step {step}/{len(train_loader)} Loss: {loss.item():.4f}")

        epoch_loss = running_loss / total_samples
        average_dice_coefficient = total_dice_coefficient / len(
            train_loader)  # Compute the average Dice coefficient for the epoch

        print(f"Epoch {epoch + 1} Average Loss: {epoch_loss:.4f}")
        print(f"Epoch {epoch + 1} Average Dice Coefficient: {average_dice_coefficient:.4f}\n")

    torch.save(model.state_dict(), "model_checkpoint.pth")


if __name__ == "__main__":
    train()
