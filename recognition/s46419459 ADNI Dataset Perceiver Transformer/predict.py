from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cycler

from PIL import Image
import os
import random as rand

from modules import *

#path = r"C:\Users\dcp\Documents\OFFLINE-Projects\DATASETS\ADNI"  # Laptop Path
path = r"C:\Users\deepp\Documents\Offline Projects\ML Datasets\ADNI" # PC path

# Model Parameters
image_shape = (240, 240)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
d_latent = 256
embed_dim = 128
transformer_depth = 1
num_heads = 1
n_perceiver_blocks = 6
num_epochs = 50
batch_size = 5
n_classes = 2
lr = 0.005

if __name__ == "__main__":
    
    # Setup image transforms to feed into model
    transforms = transforms.Compose([
        transforms.CenterCrop(240), 
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Lambda(lambda x: torch.flatten(x, start_dim = 1))
    ])

    # Load model 
    model = Perceiver(
        d_latent,
        embed_dim,
        num_heads,
        transformer_depth,
        n_perceiver_blocks,
        n_classes,
        batch_size
    ).to(device = device)
    model.load_state_dict(torch.load("./model.pth"))
    model.eval()

    # Open a random NC image and try predict it with model
    nc_image = Image.open(path + "/test/NC/" + rand.choice(os.listdir(path + "/test/NC/")))
    nc_tensor = transforms(nc_image).to(device)
    nc_tensor = nc_tensor.view(1, 1, 240 * 240).repeat(batch_size, 1, 1)
    nc_output = model(nc_tensor)
    nc_probs = F.softmax(nc_output[0], dim = 0)
    nc_class = torch.argmax(nc_output[0], dim = 0).item()

    # Open a random NC image and try predict it with model
    ad_image = Image.open(path + "/test/AD/" + rand.choice(os.listdir(path + "/test/AD/")))
    ad_tensor = transforms(ad_image).to(device)
    ad_output = model(nc_tensor)
    ad_probs = F.softmax(nc_output[0], dim = 0)
    ad_class = torch.argmax(nc_output[0], dim = 0).item()

    nc_class_name = "NC" if nc_class == 0 else "AD"
    ad_class_name = "NC" if ad_class == 0 else "AD"

    # Display the images and their prediction probabilities
    fig, axs = plt.subplots(ncols = 2, figsize = (15,8))

    # Plotting the NC Image
    axs[0].imshow(nc_image)
    axs[0].set_title(f"NC Brain scan predicted as {nc_class_name} with prob: {nc_probs[nc_class]:.4f}")
    axs[0].axis('off')

    # Plotting the AD Image
    axs[1].imshow(ad_image)
    axs[1].set_title(f"AD Brain scan predicted as {ad_class_name} with prob: {ad_probs[ad_class]:.4f}")
    axs[1].axis('off')
    plt.show()  