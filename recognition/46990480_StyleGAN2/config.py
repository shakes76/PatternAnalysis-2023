'''
A global configuration file that drives the dataset.py, train.py & predict.pt
'''
import torch

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path Parameters
modelName = "OASIS_With_Preprocessing_A100_512_lat"
# data_path_root = "./data/OASIS/" # Rangpur
data_path_root = "/Datasets/keras_png_slices_data/" # Desktop
output_path = "./"
save_path = "./Models/"

# Training Hyper Parameters
num_epochs = 2
learning_rate = 0.0001  # The learnign Rate used during training
channels = 3  # Number of channels (rgb images, so we have 3 channels in this case)
batch_size = 64  # Training Batch Size
image_size = 64  # Spatial size of the training images - OASIS 256px
log_resolution = 4 # for 256*256 image size
image_height = 2**log_resolution
image_width = 2**log_resolution
z_dim = 256  # in the original paper these were initialised to 512
w_dim = 256  # style vector dimension
lambda_gp = 10  # Lambda value for the gradient penalty