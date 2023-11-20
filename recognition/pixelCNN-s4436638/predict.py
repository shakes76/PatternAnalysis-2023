import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from modules import pixelCNN
from dataset import GetADNITest
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from argparse import ArgumentParser

### We will use this to store the images to
if not os.path.exists("test_images/recon"):
    os.makedirs("test_images/recon")
if not os.path.exists("test_images/downscale"):
    os.makedirs("test_images/downscale")
if not os.path.exists("test_images/ground_truth"):
    os.makedirs("test_images/ground_truth")

# Get arguments about opt
parser = ArgumentParser(description='pixelCNN')
parser.add_argument('--data_path', type=str, default="/home/Student/s4436638/Datasets/AD_NC/test/", help='folder containing test images')
args = parser.parse_args()

# Get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Define the path to the dataset
images_path = args.data_path + "/"

### Define a few testing parameters
batch_size = 1
upscale_factor = 4
channels = 1
feature_size = 32
num_convs = 3
image_size_x = 256
image_size_y = 240

# Define the transform to downscale the images
down_sampler = T.Resize(size=[image_size_y // upscale_factor, image_size_x // upscale_factor], 
                    interpolation=T.transforms.InterpolationMode.BICUBIC, antialias=True)

# Define our test dataset
test_set = GetADNITest(images_path)
# Define our test dataloader
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, num_workers=4, shuffle=False)

# Print out some information about the datasets
print("Num Test: " + str(test_set.arrLen))

### Load the model
model = pixelCNN(upscale_factor, channels, feature_size, num_convs)
# Send the model to the device
model = model.to(device)
# Load weights
model.load_state_dict(torch.load("pixelCNN.pkl"))    

psnr_arr = []
ssim_arr = []
mse_arr = []

### Perform testing
with torch.no_grad():
    model.eval()

    for i, image in enumerate(test_loader):

        # Load images from dataloader
        image = image.to(device) # [1, 1, h ,w]

        # Downscale
        input = down_sampler(image)

        # Get the model prediction
        output = model(input)

        # Calculate the loss (squeeze gets rid of batch and channel [h, w])
        im_np = np.squeeze(image.cpu().numpy())
        out_np = np.squeeze(output.cpu().numpy())
        psnr_arr.append(psnr(im_np, out_np, data_range=1.0))
        ssim_arr.append(ssim(im_np, out_np, data_range=1.0))
        mse_arr.append(mse(im_np, out_np))

        # Save the image
        save_image(output, "test_images/recon/" + str(i) + ".png")
        save_image(input, "test_images/downscale/" + str(i) + ".png")
        save_image(image, "test_images/ground_truth/" + str(i) + ".png")
        
print("Mean PSNR: " + str(np.mean(np.array(psnr_arr))))
print("Mean SSIM: " + str(np.mean(np.array(ssim_arr))))
print("Mean MSE: " + str(np.mean(np.array(mse_arr))))

### Save the metrics as a .csv file
np.savetxt("psnr_loss.csv", psnr_arr, delimiter=",")
np.savetxt("ssim_loss.csv", ssim_arr, delimiter=",")
np.savetxt("mse_loss.csv", mse_arr, delimiter=",")



