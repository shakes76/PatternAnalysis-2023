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
from train import *
#Runs the training process for VQVAE and DCGAN, saving respective models
# run_training()
p = Parameters()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Function to save an image of Gan output
generated_images = gan_generated_images(device, p)

#Function to save visualisation of generated code indice
code_indice = gan_create_codebook_indice(generated_images)

#Function for decoding the generated outputs and save as final reconstruction
decoded = gan_reconstruct(code_indice)

#Calculate the average and max SSIM against test data set and print to terminal
SSIM(decoded)



