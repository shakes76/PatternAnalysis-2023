import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset import ISICDataset
from modules import ImprovedUNet, device
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def combined_loss(pred, target):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    dice = dice_loss(pred, target)
    return bce + dice


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
    # Define transformations
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.ToTensor()

    # Load dataset
    full_dataset = ISICDataset("ISIC2018_Task1-2_Training_Input_x2", "ISIC2018_Task1_Training_GroundTruth_x2", image_transform, mask_transform)

    # Split dataset
    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # Number of training epochs
    num_epochs = 15

    # Lists to keep track of training progress
    train_losses, val_losses = [], []
    avg_train_dscs, avg_val_dscs = [], []

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")  
        model.train()
        running_loss, running_dsc = 0.0, 0.0

        # Training step
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            dsc = 1 - dice_loss(outputs, labels).item()
            running_dsc += dsc
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_dsc = running_dsc / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_train_dscs.append(avg_train_dsc)

        # Validation step
        model.eval()
        val_loss, val_dsc = 0.0, 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = combined_loss(outputs, masks)
                val_loss += loss.item()
                dice_val = 1 - dice_loss(outputs, masks).item()
                val_dsc += dice_val

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dsc = val_dsc / len(val_loader)
        val_losses.append(avg_val_loss)
        avg_val_dscs.append(avg_val_dsc)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train DSC: {avg_train_dsc:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Val DSC: {avg_val_dsc:.4f}")
        scheduler.step(avg_val_loss)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, '-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(range(1, num_epochs + 1), avg_train_dscs, '-o', label='Training DSC')
    plt.title('Training Dice Similarity Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('DSC')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(range(1, num_epochs+ 1), val_losses, '-o', label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(range(1, num_epochs + 1), avg_val_dscs, '-o', label='Validation DSC')
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
