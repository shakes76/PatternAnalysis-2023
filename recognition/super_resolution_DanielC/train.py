"""
Reference: https://keras.io/examples/vision/super_resolution_sub_pixel/
"""

from modules import *
from utils import *
from dataset import *
import torch
import time
import matplotlib.pyplot as plt

"""
Train the super-resolution network using the ADNI traning data.

Attributes:
    - device:       "cuda" if nvidia gpu is present, else "cpu". 
                    Warning will be printed if gpu is not present.
    - model:        Super-resolution network as defined in modules.py.
    - train_loader: DataLoader containing ADNI training data.
    - test_loader:  DataLoader containing ADNI testing data.
    - criterion:    Mean Squared Error loss.
    - Optimizer:    Adam optimizer with learning rate of 0.001.
"""

# -------
# Initialise device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

model = SuperResolution().to(device)

# -------
# Generate dataloaders
train_loader = generate_train_loader()
test_loader = generate_test_loader()

# model info
print("Model No. of Parameters:", sum([param.nelement() 
                                       for param in model.parameters()]))
print(model)

# -------
# Train the model
total_step = len(train_loader)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# -------
# Value trackers
loss_values = []
epochs = []

model.train()
print("> Training")
start = time.time()

iteration = 1
for epoch in range(num_epochs):
    epochs.append(epoch)
    loss_val = 0
    for i, (images, _) in enumerate(train_loader):

        low_res_images = resize_tensor(images)
        low_res_images = low_res_images.to(device)
        images = images.to(device)
        # Forward pass
        outputs = model(low_res_images)
        loss = criterion(outputs, images)

        # Backward propogration and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # iteration += 1
        loss_val += loss.item()
        if (i+1) % 10 == 0:
            
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, 
                            total_step, loss.item()), flush=True)
            
    loss_values.append(loss_val)
            

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " 
      + str(elapsed/60) + " mins in total") 

# New canvas for graph
plt.figure()

plt.plot(epochs, loss_values)
plt.title("Total loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# Save the model if it performed better than previous models
if loss_values[-1] < min_loss:
    torch.save(model, model_path)
