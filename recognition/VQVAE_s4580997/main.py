import torch
from modules import VQVAE, Discriminator, Generator
from utils import VQVAEConfig, GANConfig
from dataset import Dataset
from train import TrainVQVAE

if __name__ == '__main__':
    config = VQVAEConfig()
    vqvae_dataset = Dataset(batch_size=config.batch_size, root_dir = './AD_NC', fraction=0.1)
    print('Dataset loaded, beginning training.')
    vqvae = VQVAE()
    vqvae_trainer = TrainVQVAE(vqvae, vqvae_dataset, config)
    vqvae_trainer.train()
    vqvae_trainer.plot_loss(save=True)
    # vqvae.trainer.save()

    # gan_dataset = Dataset(batch_size=config.batch_size, root_dir = './AD_NC', fraction=0.1)
    # gan = GAN()
    # gan_trainer = TrainGAN(vqvae, gan_dataset, config)
    # gan_trainer.plot_loss(save=True)
    # gan_trainer.save()
