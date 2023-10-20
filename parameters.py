###################################
# PARAMETER CONSTANTS FOR PROJECT #
###################################

# Project directory root
root = "."

# Normalizing dataset parameter
norm_mean, norm_sd = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Number of epochs in training
epochs = 150

# Batch size for loader
batch_size = 32

# Learning Rate
lr = 0.001

# Beta1 hyper-parameter for Adam optimizer
beta1 = 0.5

# Image size (nxn)
image_size = 128

# Number of channels for image
num_channels = 3

# ADaIN MODEL PARAMETERS #
# Epsilon constant in init
eps = 1e-5

# MAPPING NETWORK PARAMETERS #
# Number of layers to distribute to
num_layers = 4

# Latent space dimension (size of style z vector)
z_dim = 128

# Dimension of the hidden layer of the mapping network
hidden_dim = 512

# StyleGAN NETWORK PARAMETERS #
# Initial number of channels
init_channels = 128

# Initial resolution to be scaled
init_resolution = 4

######
# Discriminator MODEL PARAMETERS
######
# Size of the feature map
feature_map_size = 64
