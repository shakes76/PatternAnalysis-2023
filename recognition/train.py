import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from dataset import ISICDataset
from modules import ImprovedUNet
from torchvision import transforms
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice


def evaluate_dsc(loader, model, device):
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

    full_dataset = ISICDataset("ISIC2018_Task1-2_Training_Input_x2", "ISIC2018_Task1_Training_GroundTruth_x2", image_transform, mask_transform)

    train_size = int(0.85 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 3
    train_losses = []

    for epoch in range(num_epochs):
        print(f"--- Epoch {epoch + 1}/{num_epochs} ---")

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
        average_dice_coefficient = total_dice_coefficient / len(train_loader)

        print(f"Training Loss: {epoch_loss:.4f}, Training Dice Coefficient: {average_dice_coefficient:.4f}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {epoch_loss:.4f}, Training Dice Coefficient: {average_dice_coefficient:.4f}")

    # Evaluate DSC for each image in the test set
    test_dscs = evaluate_dsc(test_loader, model, device)
    print(f"\nAverage Test DSC: {sum(test_dscs) / len(test_dscs):.4f}")
    below_threshold = [d for d in test_dscs if d < 0.8]
    if below_threshold:
        print(f"{len(below_threshold)} out of {len(test_dscs)} test images have DSC below 0.8!")
    else:
        print("All test images have DSC above 0.8!")

    # Plotting
    epochs_range = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, '-o', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    torch.save(model.state_dict(), "model_checkpoint.pth")


if __name__ == "__main__":
    train()
