import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from modules import UNet
from dataset import get_loaders

# Hyperparameters
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 16
num_epochs = 32
num_workers = 4
image_height = 96
image_width = 128
test_dir = "data\ISIC2018_Task1-2_Test_Input"
test_out_dir = "data\ISIC2018_Task1-2_Test_Output"
train_dir = "data\ISIC2018_Task1-2_Training_Input_x2"
mask_dir = "data\ISIC2018_Task1_Training_GroundTruth_x2"

def train(loader, val_loader, test_loader, model, optimizer, criterion, num_epochs):
    print("Training Start")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data, mask in loader:
            data = data.to(device)
            mask = mask.to(device)

            # Forward
            predictions = model(data)
            loss = criterion(predictions, mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        img = next(iter(test_loader))
        img = img.to(device)
        img = model(img)
        torchvision.utils.save_image(img, f"Epoch_{epoch}.png")

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}")

        check_accuracy(val_loader, model, epoch, device=device)


def check_accuracy(loader, model, epoch, device="cuda"):
    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            pred = (pred > 0.5).float()

            dice_score += (2*(pred*y).sum()) / ((pred+y).sum()+1e-8)

    print(f"Dice Score: {dice_score/len(loader)}")

    model.train()



def create_mask(loader, model, num_pics, device=device, folder="/"):
    model.eval()

    for idx, x in enumerate(loader):
        if idx > num_pics:
            break
        x = x.to(device=device)
        with torch.no_grad():
            pred = model(x)
            pred = (pred>0.5).float()
        torchvision.utils.save_image(
            pred, f"{folder}/pred_{idx}.png"
        )
        

    model.train()



def main():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_height, image_width), antialias=None)
    ])

    model = UNet(3, 1).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_loader, val_loader, test_loader = get_loaders(
        train_dir,
        mask_dir,
        test_dir,
        batch_size,
        train_transforms,
    )


    train(train_loader, val_loader, test_loader, model, optimizer, criterion, num_epochs)
        
    torch.save(model.state_dict(), "model")
    create_mask(test_loader, model, 2, device=device, folder=test_out_dir)

if __name__ == "__main__":
    main()



