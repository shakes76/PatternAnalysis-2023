import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import DDPM_UNet, UNet
from dataset import ADNIDataset

from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure the size is consistent
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1] for better training stability
])

dataset = ADNIDataset(root_dir="./AD_NC",  train=True, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


