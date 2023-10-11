"""
Contains the training and validation steps for each of the models
"""
import torch
from torch import optim
from dataset import generateDataLoader
from modules import Generator, Discriminator, MappingNetwork, PathLengthPenalty

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path Parameters
modelName = "256_train_Wpre_256_lat"
# data_path_root = "./data/OASIS/" # Rangpur
data_path_root = "/Datasets/keras_png_slices_data/" # Desktop
output_path = "./"
save_path = "./Models/"

# Training Hyper Parameters
num_epochs = 2
learning_rate = 0.0001
channels = 3  # Number of channels (rgb images, so we have 3 channels in this case)
batch_size = 64  # Training Batch Size
image_size = 64  # Spatial size of the training images - OASIS 256px
log_resolution = 8 # for 256*256 image size
image_height = 2**log_resolution
image_width = 2**log_resolution
z_dim = 256  # in the original paper these were initialised to 512
w_dim = 256  # style vector dimension
lambda_gp = 10  # Lambda value for the gradient penalty

# ----------------
# Data
print("> Loading Dataset")
trainset, train_loader, *otherLoaders = generateDataLoader(image_height, image_width, batch_size, data_path_root)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Data Loader
total_steps = len(train_loader)
print("> Dataset Ready")

# ----------------
# Models
print("> Configuring Models")
print("> Training Model: " + modelName)
generator = Generator(log_resolution, w_dim).to(device)
discriminator = Discriminator(log_resolution).to(device)
mapping_network = MappingNetwork(z_dim, w_dim).to(device)
path_length_penalty = PathLengthPenalty(0.99).to(device)
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=learning_rate, betas=(0.0, 0.99))

# Generator & Discriminator info
print("Generator No. of Parameters:", sum([param.nelement() for param in generator.parameters()]))
print(generator)
print("Discriminator No. of Parameters:", sum([param.nelement() for param in discriminator.parameters()]))
print(discriminator)
print("Mapping Network No. of Parameters:", sum([param.nelement() for param in mapping_network.parameters()]))
print(mapping_network)

# ----------------
# Training the model
# Put the models into training mode
generator.train()
discriminator.train()
mapping_network.train()
