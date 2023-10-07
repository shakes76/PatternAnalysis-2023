import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataset import ISICDataset
from modules import ImprovedUNet
from torchvision import transforms


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


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
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1  
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            outputs = model(images)
            loss = dice_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)  # multiply by batch size
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        print(f"Epoch {epoch + 1}/{num_epochs} Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), "model_checkpoint.pth")


if __name__ == "__main__":
    train()
