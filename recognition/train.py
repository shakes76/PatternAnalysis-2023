import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from modules import get_maskrcnn_model
from dataset import ISICDataset
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = get_maskrcnn_model(num_classes=2)  # 1 class (lesion) + 1 background
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# path to train images
img_train_dir = 'ISIC-2017_Training_Data'
# path to train masks
mask_train_dir = 'ISIC-2017_Test_v2_Part1_GroundTruth'

train_dataset = ISICDataset(img_dir=img_train_dir, mask_dir=mask_train_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the number of epochs
epochs = 10

# Training loop
losses = []
for epoch in range(epochs):
    for images, targets in train_loader:
        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss.backward()
        optimizer.step()
        losses.append(loss.item())


# path to save model
path_to_saved_model = 'C:\\Users\\yangj\\Desktop\\COMP3710_Project\\Save_Model\\mask_rcnn_model.pth'
torch.save(model.state_dict(), path_to_saved_model)

# Plotting losses
plt.plot(losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
