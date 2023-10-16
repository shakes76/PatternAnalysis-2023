import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from modules import UNet
from dataset import SkinDataset, get_loaders

# Hyperparameters
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
num_epochs = 100
num_workers = 2
image_height = 510
image_width = 384
train_dir = "recognition\ISIC Improved UNet 47479049\data\ISIC2018_Task1-2_Training_Input_x2"
mask_dir = "recognition\ISIC Improved UNet 47479049\data\ISIC2018_Task1_Training_GroundTruth_x2"

def train(loader, model, optimizer, criterion, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device=device)

        # Forward
        predictions = model(data)
        loss = criterion(predictions, targets)


        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # tqdm
        loop.set_postfix(loss=loss.item())



def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            ToTensorV2(),
        ],
    )   

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    train_loader = get_loaders(
        train_dir,
        mask_dir,
        batch_size,
        train_transforms,
        val_transforms,
    )

    for epoch in range(num_epochs):
        train(train_loader, model, optimizer, criterion, scaler)




if __name__ == "__main__":
    main()



