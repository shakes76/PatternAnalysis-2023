import torch
from modules import VQVAE, Discriminator, Generator
from utils import Config
from dataset import Dataset
from train import TrainVQVAE

if __name__ == '__main__':
    config = Config()
    vqvae_dataset = Dataset(batch_size=config.batch_size, root_dir = './AD_NC', fraction=0.1)
    print('Dataset loaded, beginning training.')
    vqvae = VQVAE()
    vqvae_trainer = TrainVQVAE(vqvae, vqvae_dataset, config)
    vqvae_trainer.train()
    print('Model trained, beginning testing.')
    vqvae_trainer.plot_loss()