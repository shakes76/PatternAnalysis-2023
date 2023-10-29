'''contains the hyperparameters and path config'''



# Path
DATA = "/Users/4vir4l/dev/data/keras_png_slices_data"

# Hyper Parameters
epochs = 300            # Number of epochs to train
learning_rate = 1e-3    # Learning rate
channels = 3            # Number of channels (3 channels as the image is RGB)
batch_size = 32         # Batch Size
image_size = 64         # Spatial size of the images - OASIS 256px
log_resolution = 7      # 256*256 image size as such 2^8 = 256
image_height = 2**log_resolution    # The height of the generated image
image_width = 2**log_resolution     # The width of the generated image
z_dim = 256             # Size of the z latent space [initialise to 256 for lower VRAM usage or faster training]
w_dim = 256             # Size of the style vector latent space [initialise to 256 for lower VRAM usage or faster training]
lambda_gp = 10          # WGAN-GP set to standard value 10