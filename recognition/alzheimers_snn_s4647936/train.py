import os
import torch
import matplotlib.pyplot as plt
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

# Test to see number of images
print(len(train_dataset)) # 21520
print(len(test_dataset)) # 9000

# Get a sample triplet from the training dataset
anchor, positive, negative = train_dataset[0]

# Now check for NaN and Inf values
print(torch.isnan(anchor).any())
print(torch.isinf(anchor).any())

# Function to display images
def imshow(img):
    img = img.numpy().transpose((1, 2, 0))
    
    # Convert to float and normalize if necessary
    if img.max() > 1:
        img = img.astype(float) / 255
        
    plt.imshow(img)
    plt.show(block=True)

# Display the images
imshow(anchor)
imshow(positive)
imshow(negative)

# Example to save the image
plt.savefig('sample_image.png')

