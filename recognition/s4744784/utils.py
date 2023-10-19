"""
File for all utilities/constants used in the model/other files.
"""
import os
import torch
import torchvision.transforms as transforms
from torch.nn.modules.upsampling import Upsample

# CONSTANTS
# HYPERPARAMETERS
upscale_factor = 4
num_epochs = 100
learning_rate = 0.001
channels = 1
dropout_probability = 0.3

image_width = 256
image_height = 240
downscale_factor = 4
new_width = image_width // downscale_factor
new_height = image_height // downscale_factor
upscale_factor = 4
batch_size = 8

ngpu = torch.cuda.device_count() # number of GPUs available. Use 0 for CPU mode.
num_workers = 2 * ngpu if ngpu > 1 else 2 # number of subprocesses to use for data loading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# PATHS FOR LOCAL DEVELOPMENT
directory = os.path.abspath('./data/AD_NC')
train_path = os.path.join(directory, 'train')
test_path = os.path.join(directory, 'test')

trained_path = os.path.abspath('./models/Trained_Model_Epoch_100.pth')

down_sample = transforms.Compose([transforms.Resize((new_height, new_width), antialias=True)])
up_sample = Upsample(scale_factor=upscale_factor)

def compute_psnr(mse, max_pixel_val=1.0):
    mse_tensor = torch.tensor(mse).to(device)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(max_pixel_val**2 / mse_tensor)