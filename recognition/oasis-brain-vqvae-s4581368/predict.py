"""
Prediction framework for OASIS brain generation Task 8
Ryan Ward
45813685
"""
import torch
from torchvision.utils import make_grid
from modules import VQVAE
from dataset import data_loaders
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse
from constants import *

def generate_vqvae_samples(model, test_loader, device):
    """
    Generate reconstructed samples from the VQ-VAE
    :param nn.Module model: The VQ-VAE model
    :param Dataloader test_loader: The dataset to reconstruct images from 
    :param str device: The device to generate the models on
    """
    with torch.no_grad():
        images = next(iter(test_loader))
        images = images.to(device)
        _, x_tilde, _, _ = model(images)

    return x_tilde

def show_quantized_image(test_data, model, device):
    """
    Show the discrete represantation of a given image from the VQ-VAE
    :param Dataloader test_loader: The dataset to reconstruct images from 
    :param nn.Module model: The VQ-VAE model
    :param str device: The device to generate the models on
    """
    with torch.no_grad():
        fixed_images = next(iter(test_data))
        fixed_images = fixed_images.to(device)

        # Get the quantized represantations
        _, _, quantized, _ = model(fixed_images)

        # Get the first image, convert to a numpy array
        q = quantized[0][0].cpu()
        q = q.detach().numpy()
        plt.imshow(q)
        plt.show()

def save_losses(recon_losses, ssims, gen_loss, discrim_loss):
    """
    Method to save losses and ssims from training
    :param list recon_losses: reconstruction losses from the VQ-VAE
    :param list ssims: A list of SSIMS from training
    :param list gen_loss: The generator losses form training
    :param list discrim_loss: The discriminator loss from training
    """
    plt.figure()
    plt.plot(recon_losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("./logs/reconstruction_error_training.png")
    plt.figure()
    plt.plot(ssims)
    plt.xlabel("EPOCH")
    plt.ylabel("SSIM")
    plt.savefig("./logs/training_ssims.png")
    plt.figure()
    plt.plot(gen_loss)
    plt.xlabel("EPOCH")
    plt.ylabel("Generator Loss")
    plt.savefig("./logs/gen_loss.png")
    plt.figure()
    plt.plot(discrim_loss)
    plt.xlabel("EPOCH")
    plt.ylabel("Discriminator Loss")
    plt.savefig("./logs/discrim_loss.png")

def generate_samples(latent_tensors, gan, vqvae):
    """
    Helper function to save generated MRI images
    :param Tensor latent_tensors: The latent tensors to input to the GAN
    :param nn.Module gan: The trained GAN model to generate the discrete represantations
    :param nn.Module vqvae: The trained VQ-VAE to generate novel MRI images from the generator output
    """
    # Get a fake batch of discrete representations
    fake_images = gan.generator(latent_tensors)
    # Feed discrete vectors into the VQ-VAE decoder
    fake_images = vqvae.decoder(fake_images) 
    # Show the images
    plt.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
    plt.show()

def ssim_batch(x, x_tilde):
    """
    Calculate the SSIM for each image in a batch
    :param Tensor x: The original images
    :param Tensor x_tilde: The reconstructed images
    """
    ssims = []
    for i in range(x.shape[0]):
        calculated_ssim = ssim(x[i, 0], x_tilde[i, 0], data_range=(x_tilde[i, 0].max() - x_tilde[i, 0].min()))
        ssims.append(calculated_ssim)
    return ssims

def calculate_batch_ssim(data_loader, model, device):
    """
    Calculate the average SSIM for a batch of images
    :param Dataloader data_loader: The original image data
    :param nn.Module model: The VQ-VAE model in training or trained
    :param str device: The device to generate on
    """
    with torch.no_grad():
        images = next(iter(data_loader))
        images = images.to(device)
        _, x_tilde, _, _= model(images)
        x_tilde = x_tilde.cpu().detach().numpy()
        images = images.cpu().detach().numpy()
        calculated_ssims = ssim_batch(images, x_tilde)
        avg_ssim = sum(calculated_ssims)/len(calculated_ssims)
    return avg_ssim
 
def main(args):
    """
    This module can be run by itself, taking in a trained VQ-VAE and GAN
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_path= "./keras_png_slices_data/keras_png_slices_test"
    vq_path = args.vq_path
    vq_vae = VQVAE(HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA)
    vq_vae.load_state_dict(torch.load(vq_path))
    vq_vae.to(device)
    vq_vae.eval()
    _, test_loader, _ = data_loaders(test_path, test_path, test_path)
    generate_vqvae_samples(vq_vae, test_loader, device)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Specify the prediction parameters for Task 8"
    )
    parser.add_argument('vq_path', type=str, help="The VQ-VAE pretrained model to generated predictions")
    parser.add_argument('--gan_path', type=str, help="The GAN pretrained model to generated predictions")
    args = parser.parse_args()
    main(args)

