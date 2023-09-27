import torch

from dataset import *
from modules import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np

#-------------
# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Warning, CUDA not found. Using CPU.")
print()


#---------------
# Hyper Parameters
learning_rate = 0.00005
num_epochs = 2
path = r"c:\Users\Jackie Mann\Documents\Jarrod_Python\AD_NC"
save_path = r"c:\Users\Jackie Mann\Documents\Jarrod_Python\PatternAnalysis-2023\recognition\super_resolution_network_s4696612\saved_model.pth"

#-----------------
# Data
batch_size = 32
train_path = path + "\\train\\AD"
test_path = path + "\\test\\AD"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_data = ImageDataset(directory=train_path,
                          transform=transform)
train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
n_total_steps = len(train_loader)

batch = next(iter(train_loader))

#----------------------------
#Training
model = SuperResolution()
model.to(device)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

print("> Training.")

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item()}")

model.eval()
torch.save(model.state_dict(), save_path)

x = batch[0].to(device)[:64]
y = model(x)
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Model Images')
plt.imshow(np.transpose(torchvision.utils.make_grid(y, padding=2,normalize=True).cpu(), (1,2,0)))
plt.show()

plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Goal Images')
plt.imshow(np.transpose(torchvision.utils.make_grid(batch[1].to(device)[:64], padding=2,normalize=True).cpu(), (1,2,0)))
plt.show()