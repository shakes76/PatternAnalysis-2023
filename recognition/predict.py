import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import ISICDataset
from modules import ImprovedUNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def predict():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = ISICDataset("ISIC2018_Task1-2_Test_Input", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ImprovedUNet().to(device)
    model.load_state_dict(torch.load("model_checkpoint.pth"))
    model.eval()

    for images, _ in test_loader:
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)


if __name__ == "__main__":
    predict()
