import os
import torch
from dataset import TripletDataset 
from torchvision import transforms

# Transformations for images
transform = transforms.Compose([
    transforms.Resize((256, 240)),
    transforms.ToTensor(),
])

# Dataset instances
train_dataset = TripletDataset(root_dir="/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/AD_NC", mode='train', transform=transform)
test_dataset = TripletDataset(root_dir="/home/Student/s4647936/PatternAnalysis-2023/recognition/alzheimers_snn_s4647936/AD_NC", mode='test', transform=transform)

# Continue training code...
