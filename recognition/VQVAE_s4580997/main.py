##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
from modules import VQVAE, GAN
from utils import VQVAEConfig, GANConfig
from dataset import Dataset, ModelDataset
from train import TrainVQVAE, TrainGAN
from predict import Predict
import PIL 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

ADNI_ROOT_DIR = './AD_NC'

# def generate_images(n_images, vqvae_model, image_size=(3, 256, 256), device='cuda'):
#     """
#     Generate images using a trained VQ-VAE model.

#     Parameters:
#     - n_images (int): Number of images to generate.
#     - vqvae_model (VQVAE): A trained VQVAE model.
#     - image_size (tuple): The size (C, H, W) of the generated images.
#     - device (str): Device to use for generating images.

#     Returns:
#     - generated_images (list of PIL.Image): List of generated images.
#     """
#     # Set VQ-VAE model to evaluation mode and to the desired device.
#     device = torch.device(device if torch.cuda.is_available() else "cpu")
#     vqvae_model.to(device)
#     vqvae_model.eval()

#     generated_images = []

#     for _ in range(n_images):
#         # Sample a random vector.
#         z = torch.randn(*image_size).to(device)

#         # Generate an image by passing the random vector through the decoder.
#         with torch.no_grad():
#             generated_tensor = vqvae_model.decoder(z.unsqueeze(0))

#         # Post-process the generated tensor to create an image.
#         generated_tensor = (generated_tensor.squeeze(0) + 1) / 2  # Scale to [0, 1]
#         generated_tensor.clamp_(0, 1)  # Ensure valid pixel values
#         generated_image = transforms.ToPILImage()(generated_tensor.cpu())

#         generated_images.append(generated_image)

#     return generated_images

def generate_images(num_images, vqvae_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae_model.to(device)
    vqvae_model.eval()  # Set the model to evaluation mode
    H = 2
    W = 2
    latent = 256
    generated_images = []

    with torch.no_grad():  # Ensure no gradients are computed
        for _ in range(num_images):
            # Generate a random latent vector
            latent_sample = torch.randn((1, latent)).to(device)

            # This latent vector might need reshaping before being passed to the decoder.
            # For example, if your decoder expects a 4D input (batch, channels, height, width),
            # you might need to reshape your latent vector appropriately.
            latent_sample = latent_sample.view(1, latent, H, W)  # Reshape
            
            # Decode the latent vector
            generated_image = vqvae_model.decoder(latent_sample)
            
            # Append to the list of generated images
            generated_images.append(generated_image)

    return generated_images


if __name__ == '__main__':
    vqvae_config = VQVAEConfig()
    gan_config = GANConfig()

    vqvae_dataset = Dataset(batch_size=vqvae_config.batch_size, root_dir = ADNI_ROOT_DIR, fraction=0.1)
    vqvae = VQVAE()
    vqvae_trainer = TrainVQVAE(vqvae, vqvae_dataset, vqvae_config)
    vqvae_trainer.train()
    vqvae_trainer.plot(save=True)
    vqvae_trainer.save(vqvae_config.model_path)

    vqvae.load_state_dict(torch.load("./models/vqvae/vqvae.pth"))

    gan_dataset = ModelDataset(vqvae, batch_size=gan_config.batch_size, root_dir = ADNI_ROOT_DIR, fraction=0.1)
    gan = GAN(features = 128, latent_size = 128)
    # gan.discriminator.load_state_dict(torch.load("./models/gan/gan_discriminator.pth"))
    # gan.generator.load_state_dict(torch.load("./models/gan/gan_generator.pth"))

    gan_trainer = TrainGAN(gan, gan_dataset, gan_config)
    gan_trainer.train()
    gan_trainer.plot(save=True)
    gan_trainer.save(gan_config.discriminator_path, gan_config.generator_path)
    
    noise = torch.randn(128, 128, 1, 1).to(gan_config.device)
    predictor = Predict(noise, n = 1, savepath='./models/predictions/output', model=gan)
    predictor.generate()
    predictor.show_generated(save=False)