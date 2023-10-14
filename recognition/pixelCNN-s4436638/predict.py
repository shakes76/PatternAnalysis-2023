import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from modules import pixelCNN
from dataset import GetADNITest
from scipy.io import savemat, loadmat

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the path to the dataset
images_path = "/home/Student/s4436638/Datasets/AD_NC/test/*"

### Define a few training parameters
batch_size = 1
upscale_factor = 4
channels = 1
feature_size = 32
num_convs = 3
learning_rate = 1e-3
image_size_x = 256
image_size_y = 240

# Define the transform to downscale the images
down_sampler = T.Resize(size=[image_size_y // upscale_factor, image_size_x // upscale_factor], 
                    interpolation=T.transforms.InterpolationMode.BICUBIC, antialias=True)

# Define our training and validation datasets
test_set = GetADNITest(images_path)
# Define our training and validation dataloaders
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4, shuffle=False)

# Print out some information about the datasets
print("Num Test: " + str(test_loader.arrLen))

### Load the model
model = pixelCNN(upscale_factor, channels, feature_size, num_convs)
# Send the model to the device
model = model.to(device)
model.load_state_dict(torch.load("pixelCNN.pkl"))    

### Perform testing
with torch.no_grad():
    model.eval()
    total_loss = 0

    for image in test_loader:

        # Load images from dataloader
        image = image.to(device)

        # Downscale
        input = down_sampler(image)

        # Get the model prediction
        output = model(input)

        # Calculate the loss
        loss = loss_function(image, output)

        # Aggregate the loss
        total_loss += loss.item()

# Divide the total loss by the number of epochs
total_loss = total_loss / val_set.arrLen
val_loss.append(total_loss)



