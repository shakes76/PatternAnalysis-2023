import torch
from modules import VQVAE, GAN
from utils import VQVAEConfig, GANConfig
from dataset import Dataset, ModelDataset
from train import TrainVQVAE, TrainGAN
from predict import Predict

ADNI_ROOT_DIR = './AD_NC'

if __name__ == '__main__':
    vqvae_config = VQVAEConfig()
    gan_config = GANConfig()

    vqvae_dataset = Dataset(batch_size=vqvae_config.batch_size, root_dir = ADNI_ROOT_DIR, fraction=0.1)
    vqvae = VQVAE()
    vqvae_trainer = TrainVQVAE(vqvae, vqvae_dataset, vqvae_config)
    vqvae_trainer.train()
    vqvae_trainer.plot(save=True)
    vqvae_trainer.save(vqvae_config.model_path)

    gan_dataset = ModelDataset(vqvae, batch_size=gan_config.batch_size, root_dir = ADNI_ROOT_DIR, fraction=0.1)
    gan = GAN(features = 128, latent_size = 128)
    gan_trainer = TrainGAN(gan, gan_dataset, gan_config)
    gan_trainer.train()
    gan_trainer.plot(save=True)
    gan_trainer.save(gan_config.discriminator_path, gan_config.generator_path)
    
    noise = torch.randn(128, 128, 1, 1).to(gan_config.device)
    predictor = Predict(gan_config.model_path, noise, n = 16, savepath='./models/predictions/output')
    predictor.generate()
    predictor.show_generated(save=True)