import os
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from modules import pixelCNN
from dataset import GetADNITrain
from scipy.io import savemat, loadmat

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the path to the dataset
images_path = "/home/Student/s4436638/Datasets/AD_NC/train/*"

### Define a few training parameters
batch_size = 10
upscale_factor = 4
channels = 1
feature_size = 32
num_convs = 3
learning_rate = 1e-3
image_size_x = 256
image_size_y = 240
num_epochs = 100

# Define the transform to downscale the images
down_sampler = T.Resize(size=[image_size_y // upscale_factor, image_size_x // upscale_factor], 
                    interpolation=T.transforms.InterpolationMode.BICUBIC, antialias=True)

# Define the loss function
loss_function = torch.nn.MSELoss()

# Define our training and validation datasets
train_set = GetADNITrain(images_path, train_split=0.9, train=True)
val_set = GetADNITrain(images_path, train_split=0.9, train=False)
# Define our training and validation dataloaders
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, num_workers=4, shuffle=True)
val_loader = DataLoader(dataset=val_set, batch_size=batch_size, num_workers=4, shuffle=False)

# Print out some information about the datasets
print("Num Train: " + str(train_set.arrLen))
print("Num Val: " + str(val_set.arrLen))

### Load the model
model = pixelCNN(upscale_factor, channels, feature_size, num_convs)
# Send the model to the device
model = model.to(device)

# Print the number of parameters in the model
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of trainable params: " + str(pytorch_total_params))

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Define some variables to keep track of the loss
train_loss = []
val_loss = []
min_val = 10000000000 # Start with the largest one
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:

        # Load images from dataloader
        images = batch.to(device)

        # Downscale
        inputs = down_sampler(images)

        # Get the model prediction
        outputs = model(inputs)

        # Calculate the loss
        loss = loss_function(images, outputs)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Aggregate the loss
        total_loss += loss.item()

    # Divide the total loss by the number of epochs
    total_loss = total_loss / train_set.arrLen
    train_loss.append(total_loss)

    # Print the loss at the end of epoch
    print_output = "[%02d/%02d] Train Loss: %.5f\n" % (epoch+1, num_epochs, total_loss)
    print(print_output)

    

    ### Perform validation
    with torch.no_grad():
        model.eval()
        total_loss = 0

        for batch in val_loader:

            # Load images from dataloader
            images = batch.to(device)

            # Downscale
            inputs = down_sampler(images)

            # Get the model prediction
            outputs = model(inputs)

            # Calculate the loss
            loss = loss_function(images, outputs)

            # Aggregate the loss
            total_loss += loss.item()

    # Divide the total loss by the number of epochs
    total_loss = total_loss / val_set.arrLen
    val_loss.append(total_loss)

    # Print the loss at the end of epoch
    print_output = "[%02d/%02d] Val Loss: %.5f\n" % (epoch+1, num_epochs, total_loss)
    print(print_output)

    # Whether or not to save the model
    if total_loss <= min_val:
        min_val = total_loss
        torch.save(model.state_dict(), "pixelCNN.pkl")



