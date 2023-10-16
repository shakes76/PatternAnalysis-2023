"""
Created on Monday Sep 18 12:20:00 2023

This script is for demonstrating an example of the trained model.
Any results and visualisations created are generated from this script.
SSIM function is also called to summarise findings.

@author: Gabriel Russell
@ID: s4640776

"""
from modules import *
from train import *
from dataset import *

def predict():
    """
    Predict function calls the training function for VQVAE and DCGAN.
    It also generates images from the trained DCGAN and other relevant 
    images. Also calls a function for calculating SSIM which is 
    printed to terminal.

    Args:
        None

    Returns:
        None
    """
    #Call training function for VQVAE and DCGAN
    VQVAE_model, Gan_loader = run_training()

    #Initialise parameters
    p = Parameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load OASIS test dataset to visualise codebook indices and quantized image
    load_data = OASISDataloader()
    visualise_VQVAE_indices(load_data, VQVAE_model, device)

    #Visualise the Gan training dataset formed from codebook indice images
    visualise_gan_loader(Gan_loader)

    #Function to save an image of Gan output
    generated_images = gan_generated_images(device, p)

    #Function to save visualisation of generated code indice
    code_indice = gan_create_codebook_indice(generated_images)

    #Function for decoding the generated outputs and save as final reconstruction
    decoded = gan_reconstruct(code_indice)

    #Calculate the average and max SSIM against test data set and print to terminal
    SSIM(decoded)

predict()


