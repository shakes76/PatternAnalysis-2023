"""
This file is used to run the created model/algorithm for predictions or testing model performance.

Author: Felix Hall
Student number: 47028643
"""

# 1. Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ADNIDataset  # Assuming the dataset is defined in a file called dataset.py
from modules import VisionTransformer  # Assuming the model is defined in a file called model.py
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import random, os



# locking in seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 2. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print ("device: ", device)
model_num = 45
batch_size = 128
img_size = 256
depth = 4
n_heads = 4
mlp_ratio = 2.0
embed_dim = 256
drop_p = 0.25
attn_p = 0.25
model_path = "saved_models/best_model_{}.pth".format(model_num)

print ("model num: ", model_num)

std = 0.2244 # from file
mean = 0.1156 # from file
data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# 3. Dataset and DataLoader

root_dir = '/home/groups/comp3710/ADNI/AD_NC'
test_dataset = ADNIDataset(root_dir=root_dir, subset='test', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 4. Model Loading
model = VisionTransformer(
    img_size=256, 
    patch_size=16, 
    in_channels=1, 
    n_classes=1, 
    embed_dim=embed_dim,
    depth=depth,  # Increased Depth
    n_heads=n_heads,  # Modified Number of Heads
    mlp_ratio=mlp_ratio,  # Modified MLP Ratio
    qkv_bias=True, 
    p=drop_p, 
    attn_p=attn_p,  # attention dropout probability
)
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# 5. Visualize Attention
def visualize_attention(data, attention_map, idx):
    # Make sure the directory exists
    os.makedirs(f'eval/{model_num}', exist_ok=True)

    # Normalize the image tensor and convert it to NumPy array
    image = data[0].squeeze().cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())

    # Select the first head's attention map for the first patch
    # Assuming you want the attention weights from the first block and the first sample in the batch
    attn_map = attention_map[0][0, 0, :, :].cpu().detach().numpy()

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Plot original image
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Original Image')

    # Plot attention map
    axs[1].imshow(attn_map, cmap='viridis')
    axs[1].set_title('Attention Map')

    plt.savefig(f'eval/{model_num}/attention_visualization_{idx}.png')
    plt.close()


# Testing the Final Model
print ("~~~ TESTING ~~~")
model.eval()  # Set the model to evaluation mode
test_loss = 0.0
correct = 0
total = 0

with torch.no_grad():  # No need to track gradients
    for idx, (data, labels) in enumerate(test_loader):
        data, labels = data.to(device), labels.to(device)
        labels = labels.view(-1, 1).float()  # Reshaping from [32] to [32, 1]

        # Forward pass
        outputs, attn_weights = model(data)

        # Visualize attention for the first batch (you can choose other batches)
        if (idx % 10 == 0) and idx < 19:
            visualize_attention(data, attn_weights, idx)
        
        # Apply sigmoid to get probabilities
        outputs = torch.sigmoid(outputs)
        
        # Thresholding
        predicted = (outputs > 0.5).float()

        # Compute the accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print the test loss and accuracy
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy}%")
