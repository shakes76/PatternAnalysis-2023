"""
Created on Monday Sep 18 12:20:00 2023

This script is for demonstrating an example of the trained model.
ANy results and visualisations created are generated from this script.

@author: Gabriel Russell
@ID: s4640776

"""
import torch
from modules import *
import numpy as np
import matplotlib.pyplot as plt


p = Parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gen_images = 32
fixed_noise = torch.randn(num_gen_images, p.channel_noise, 1, 1).to(device)

#Load Trained Generator
Generator = torch.load("Generator.pth")
Generator.eval()

with torch.no_grad():
    generated_images = Generator(fixed_noise)

# Rescale images from [-1, 1] to [0, 1] for displaying/saving
generated_images = 0.5 * (generated_images + 1)

# Convert tensor to NumPy array
generated_images = generated_images.cpu()
generated_images = generated_images.numpy()

#Display Generated images
for i in range(num_gen_images):
    plt.imshow(np.transpose(generated_images[i], (1, 2, 0)))
    plt.axis('off')
    plt.show()

