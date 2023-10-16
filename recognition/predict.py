import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset import ISICDataset  
from modules import ImprovedUNet, device  
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

print(torch.cuda.is_available())


def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def combined_loss(pred, target, alpha=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    return alpha * bce + (1 - alpha) * dice_loss(pred, target)


def evaluate_dsc(loader, model):
    model.eval()
    all_dscs = []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = dice_loss(outputs, masks)
            dice_coefficient = 1 - loss.item()
            all_dscs.append(dice_coefficient)
    return all_dscs


def train():
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    mask_transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = ISICDataset("ISIC2018_Task1-2_Training_Input_x2", "ISIC2018_Task1_Training_GroundTruth_x2",
                               image_transform, mask_transform)

    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    num_epochs = 2
    train_losses = []
    val_losses = []
    average_dscs = []
    average_val_dscs = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        val_running_loss = 0.0
        total_samples = 0
        total_val_samples = 0
        total_dice_coefficient = 0.0
        total_val_dice_coefficient = 0.0

        # Training phase
        for images, masks in train_loader:

            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            dice_coefficient = 1 - dice_loss(outputs, masks).item()
            total_dice_coefficient += dice_coefficient
            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)

                dice_coefficient = 1 - dice_loss(outputs, masks).item()
                total_val_dice_coefficient += dice_coefficient
                val_running_loss += loss.item() * images.size(0)
                total_val_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        epoch_val_loss = val_running_loss / total_val_samples
        average_dice_coefficient = total_dice_coefficient / len(train_loader)
        average_val_dice_coefficient = total_val_dice_coefficient / len(val_loader)

        train_losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        average_dscs.append(average_dice_coefficient)
        average_val_dscs.append(average_val_dice_coefficient)

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}, Training DSC: {average_dice_coefficient:.4f}")
        print(
            f"Epoch {epoch + 1}/{num_epochs} - Validation Loss: {epoch_val_loss:.4f}, Validation DSC: {average_val_dice_coefficient:.4f}")

        scheduler.step(epoch_val_loss)

    # Evaluation phase
    test_dscs = evaluate_dsc(test_loader, model)
    print(f"\nAverage Test DSC: {sum(test_dscs) / len(test_dscs):.4f}")

    # Plotting
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_losses, '-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, average_dscs, '-o', label='Training DSC')
    plt.title('Training Dice Similarity Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs_range, val_losses, '-o', label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(epochs_range, average_val_dscs, '-o', label='Validation DSC')
    plt.title('Validation Dice Similarity Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Save the model after all epochs
    torch.save(model.state_dict(), "model_checkpoint.pth")


if __name__ == "__main__":
    train()
