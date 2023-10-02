# 1. Import Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import ADNIDataset  # Assuming the dataset is defined in a file called dataset.py
from model_test import VisionTransformer  # Assuming the model is defined in a file called model.py
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import random



# locking in seed
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 2. Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print ("device: ", device)
model_num = 9
batch_size = 32
model_path = "saved_models/best_model_{}.pth".format(model_num)

print ("model num: ", model_num)

data_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])

# 3. Dataset and DataLoader

root_dir = '/home/groups/comp3710/ADNI/AD_NC'
test_dataset = ADNIDataset(root_dir=root_dir, subset='test', transform=data_transforms)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# 4. Model Loading
model = VisionTransformer()
model.load_state_dict(torch.load(model_path))
model = model.to(device)

# 5. Visualize Attention
def visualize_attention(data, attention_map):
    # Create directory if it doesn't exist
    directory = f'eval/{model_num}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    # Visualize the first image in the batch
    img = data[0].cpu().squeeze().numpy()
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')

    # Visualize the attention map for the first image in the batch
    attn_map = attention_map[0].cpu().squeeze().numpy()
    axs[1].imshow(attn_map, cmap='viridis')
    axs[1].set_title('Attention Map')

    plt.savefig(f'{directory}/attention_visualization_{idx}.png')

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
        if (idx % 10 == 0):
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
