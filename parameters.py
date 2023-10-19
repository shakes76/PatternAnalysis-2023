###################################
# PARAMETER CONSTANTS FOR PROJECT #
###################################

# Number of epochs in training
epochs = 150

# Batch size for loader
batch = 32

# Learning Rate
lr = 0.001

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
