import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import get_maskrcnn_model  # Ensure this function is defined in your modules.py
from dataset import ISICDataset  # Ensure this class is defined in your dataset.py
import matplotlib.pyplot as plt
import os

# Ensure GPU usage if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize model and optimizer
model = get_maskrcnn_model(num_classes=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Define dataset paths
img_train_dir = './ISIC2018_Task1-2_Training_Input'
mask_train_dir = './ISIC2018_Task1_Training_GroundTruth'
img_test_dir = './ISIC2018_Task1-2_Test_Input'

# Initialize datasets and dataloaders
train_dataset = ISICDataset(img_dir=img_train_dir, mask_dir=mask_train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Training parameters
epochs = 10
losses = []

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)

        # Debug: Check the loss_dict
        print("Loss Dict:", loss_dict)

        loss = sum(loss for loss in loss_dict.values())

        # Debug: Check the loss and its type
        print("Loss:", loss, "Type:", type(loss))

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

# Save model
path_to_saved_model = './Save_Model'
if not os.path.exists(path_to_saved_model):
    os.makedirs(path_to_saved_model)
torch.save(model.state_dict(), os.path.join(path_to_saved_model, './Save_Model'))

# Plot losses
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
