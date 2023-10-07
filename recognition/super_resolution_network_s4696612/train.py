import torch
from dataset import *
from modules import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.data import random_split


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
learning_rate = 0.0005
num_epochs = 1

#-----------------
# Data configuration
batch_size = 50
train_set_proportion = 0.8
valid_and_test_remaining_proportion = 0.5

# Path parameters must be changed depending on where the dataset is located on the machine
path = r"c:\Users\Jackie Mann\Documents\Jarrod_Python\AD_NC"
save_path = r"c:\Users\Jackie Mann\Documents\Jarrod_Python\PatternAnalysis-2023\recognition\super_resolution_network_s4696612\saved_model.pth"
train_path = path + "\\train\\AD"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

data = ImageDataset(directory=train_path,
                          transform=transform)
train_size = int(train_set_proportion * len(data))
remaining_size = len(data) - train_size

train_data, remaining_data = random_split(data, [train_size, remaining_size])
validation_size = remaining_size // 2
test_size = remaining_size - validation_size
test_data, validation_data = random_split(remaining_data, [test_size, validation_size])

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                          batch_size=batch_size,
                                          shuffle=True)
valid_loader = torch.utils.data.DataLoader(validation_data,
                                           batch_size=batch_size,
                                           shuffle=True)
n_total_steps = len(train_loader)
batch = next(iter(train_loader))

#----------------------------
#Training
model = SuperResolution()
model.to(device)
model.train()

# Adam optimiser used for learning
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Mean squared error loss used for image comparisons
criterion = nn.MSELoss()

# Learning rate scheduler used to vary learning
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.1)

print("> Training.")

losses = []

# Training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item()}")
    scheduler.step()

#-------------------
# Model Finalisation
model.eval()

# Save trained pytorch model
torch.save(model.state_dict(), save_path)

# Show sample of model images
x = batch[0].to(device)[:64]
y = model(x)
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Model Images')
plt.imshow(np.transpose(torchvision.utils.make_grid(y, padding=2,normalize=True).cpu(), (1,2,0)))
plt.show()

# Show sample of goal images
plt.figure(figsize=(8,8))
plt.axis('off')
plt.title('Goal Images')
plt.imshow(np.transpose(torchvision.utils.make_grid(batch[1].to(device)[:64], padding=2,normalize=True).cpu(), (1,2,0)))
plt.show()

plt.figure()
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error Loss')
plt.plot(losses)
plt.savefig('training_loss.png')
plt.show()