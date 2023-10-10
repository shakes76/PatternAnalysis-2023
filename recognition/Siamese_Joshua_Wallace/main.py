import torch
from modules import VQVAE, Discriminator, Generator
from utils import Config
from dataset import ADNIDataset, GANDataset
from train import VQVAETrain, GANTrain

if __name__ == '__main__':
    config = Config(lr=1e-4, wd=1e-5, epochs=1, batch_size=32, root_dir='./AD_NC', gpu='cuda')
    vqvae_dataset = ADNIDataset(batch_size=config.batch_size, fraction=0.1)
    gan_dataset = GANDataset(batch_size=config.batch_size, fraction=0.1)
    print('Dataset loaded, beginning training.')
    VQVAE = VQVAE()

    vqvae_trainer = VQVAETrain(net, dataset, config)
    vqvae_trainer.train()
    print('Model trained, beginning testing.')
    vqvae_trainer.test()
    vqvae_trainer.plot_loss()