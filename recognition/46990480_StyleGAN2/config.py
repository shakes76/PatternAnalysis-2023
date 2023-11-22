'''
A global configuration file that drives the dataset.py, train.py & predict.pt
'''
import torch

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path Parameters
modelName = "OASIS_With_Preprocessing_512_latent"   # Name of the model
data_path_root = "/Datasets/keras_png_slices_data/" # Desktop
output_path = "./"                                  # The root output path (used for saved example outputs)
save_path = "./Models/"                             # The save path, used to save the trained models

# Training Hyper Parameters
num_epochs = 300        # Number of epochs to train
learning_rate = 0.0001  # The learning rate used during training
channels = 3            # Number of channels (rgb images, so we have 3 channels in this case)
batch_size = 64         # Training Batch Size
image_size = 64         # Spatial size of the training images - OASIS 256px
log_resolution = 8      # for 256*256 image size
image_height = 2**log_resolution    # The height of the images you wish to generate
image_width = 2**log_resolution     # The width of the image you wish to generate
z_dim = 512             # Size of the z latent space
w_dim = 512             # Size of the style vector latent space
lambda_gp = 10          # Lambda value for the gradient penalty WGAN_GP_LOSS calculation (as per paper)