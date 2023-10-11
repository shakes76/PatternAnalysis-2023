import torch
from modules import VQVAE, GAN
from utils import VQVAEConfig, GANConfig
from dataset import Dataset
from train import TrainVQVAE, TrainGAN

if __name__ == '__main__':
    vqvae_config = VQVAEConfig()
    gan_config = GANConfig()

    vqvae_dataset = Dataset(batch_size=vqvae_config.batch_size, root_dir = './AD_NC', fraction=0.1)
    print('Dataset loaded, beginning training.')
    vqvae = VQVAE()
    vqvae_trainer = TrainVQVAE(vqvae, vqvae_dataset, vqvae_config)
    vqvae_trainer.train()
    vqvae_trainer.plot(save=True)
    # vqvae.trainer.save()

    gan_dataset = Dataset(batch_size=gan_config.batch_size, root_dir = './AD_NC', fraction=0.1)
    gan = GAN(features = 128, latent = 128)
    gan_trainer = TrainGAN(gan, gan_dataset, gan_config)
    gan_trainer.plot(save=True)
    # gan_trainer.save()
