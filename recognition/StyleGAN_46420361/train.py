import torch
from dataset import load_dataset


# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("WARNING: CUDA not found, using CPU")

# hyperparameters
start_image_size = 8
image_size = 64
path = '~/OASIS_data/keras_png_slices_data'
learning_rate = 1e-3
batch_sizes = [256, 128, 64, 32, 16, 8]
channels = 3
z_dim = 256
w_dim = 256
in_channels = 256
lambda_gp = 10
progressive_epochs = [30]*len(batch_sizes)

dataloader, dataset = load_dataset(image_size, batch_sizes, path)