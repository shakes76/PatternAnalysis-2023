import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from modules import pixelCNN
from dataset import GetADNITrain
import matplotlib.pyplot as plt
from argparse import ArgumentParser

# Get arguments about opt
parser = ArgumentParser(description='pixelCNN')
parser.add_argument('--data_path', type=str, default="/home/Student/s4436638/Datasets/AD_NC/train/", help='folder containing test images')
args = parser.parse_args()

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the path to the dataset
images_path = args.data_path + "/"

### Define a few training parameters
batch_size = 10
upscale_factor = 4
channels = 1
feature_size = 32
num_convs = 3
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define some variables to keep track of the loss
train_loss = [] # Keep track of loss after each epoch
val_loss = []
min_val = 10000000000 # Start with the largest one
for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:

        # Load images from dataloader
        ground_truth = batch.to(device)

        # Downscale
        low_rez = down_sampler(ground_truth)

        # Get the model prediction
        prediction = model(low_rez)

        # Calculate the loss
        loss = loss_function(ground_truth, prediction)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Aggregate the loss
        total_loss += loss.item()

    # Divide the total loss by the number of images
    total_loss = total_loss / (train_set.arrLen / batch_size)
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
            ground_truth = batch.to(device)

            # Downscale
            low_rez = down_sampler(ground_truth)

            # Get the model prediction
            prediction = model(low_rez)

            # Calculate the loss
            loss = loss_function(ground_truth, prediction)

            # Aggregate the loss
            total_loss += loss.item()

    # Divide the total loss by the number of epochs
    total_loss = total_loss / (val_set.arrLen / batch_size)
    val_loss.append(total_loss)

    # Print the loss at the end of epoch
    print_output = "[%02d/%02d] Val Loss: %.5f\n" % (epoch+1, num_epochs, total_loss)
    print(print_output)

    # Whether or not to save the model
    if total_loss <= min_val:
        min_val = total_loss
        torch.save(model.state_dict(), "pixelCNN.pkl")

    ### Save the metrics as a .csv file
    np.savetxt("train_loss.csv", train_loss, delimiter=",")
    np.savetxt("val_loss.csv", val_loss, delimiter=",")

# Plot the training and validation losses
x = range(num_epochs)

plt.figure(1)
plt.plot(x, train_loss, label="Train Loss")
plt.plot(x, val_loss, label="Validation Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Training and Validation Loss")
plt.savefig("training-plot.png")
