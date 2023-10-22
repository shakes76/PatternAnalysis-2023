"""
Driver file for OASIS brain generation
Ryan Ward
45813685
"""
import os
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from datetime import datetime
from modules import VQVAE, GAN
from dataset import data_loaders
from train import train_vqvae, train_gan
from predict import save_losses, generate_samples, show_quantized_image, calculate_batch_ssim
from constants import *

def main():
    """
    The main entry point to the OASIS VQ-VAE Task 8 project. Trains the models,
    shows and saves losses, and generates novel samples
    """
    # Find device and print
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Used to save a run to a new directory
    now = datetime.now()
    time_rep = now.strftime("%m-%d-%Y-%H_%M_%S")
    logger = SummaryWriter("./logs/{0}".format(time_rep))
    save_filename = "./models/{0}".format(time_rep)
    os.makedirs(save_filename, exist_ok=True)

    # Latent tensors for the GAN
    latent_fixed = torch.rand(64, 128, 1, 1, device=device)

    # Load the data
    train_loader, test_loader, val_loader = data_loaders(TRAIN_PATH, TEST_PATH, VAL_PATH)

    # Set some fixed images to display the data
    fixed_images = next(iter(test_loader))

    # VQ-VAE definition
    vqvae = VQVAE(HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA)

    # Train the VQ-VAE
    recon_loss, avg_ssims = train_vqvae(vqvae, save_filename, device, train_loader, val_loader, logger, fixed_images, LEARNING_RATE, EPOCHS) 
    print("VQ-VAE Final SSIM: {}".format(calculate_batch_ssim(test_loader, vqvae, device)))
   
    # Simple GAN
    gan = GAN()

    # Train the GAN, with the quantization output from the VQ-VAE being fed into the gan Generator
    loss_gen, loss_discrim = train_gan(vqvae, gan, train_loader, device, save_filename, EPOCHS, latent_fixed, LEARNING_RATE)

    # Save and show results
    save_losses(recon_loss, avg_ssims, loss_gen, loss_discrim)
    show_quantized_image(test_loader, vqvae, device)
    generate_samples(latent_fixed, gan, vqvae)

if __name__ == "__main__":
    main()
