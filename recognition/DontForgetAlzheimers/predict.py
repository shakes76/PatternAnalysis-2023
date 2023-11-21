"""
Example usage of trained model
"""
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import AlzheimerDataset
from dataset import transform
from modules import ViT

from train import train

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train()
    test_dataset = AlzheimerDataset("AD_NC/test", transform=transform)
    sample_image, sample_label = test_dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(sample_image)
        _, predicted_label = torch.max(output.data, 1)

    print(f"Predicted Label: {predicted_label.item()}")
    print(f"True Label: {sample_label}")