##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
from modules import VQVAE, GAN
import utils
from dataset import Dataset, ModelDataset
from train import TrainVQVAE, TrainGAN
from predict import Predict, GenerateImages
from test import TestVQVAE
import PIL 
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from scipy.signal import savgol_filter

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vqvae =  VQVAE(channels = utils.CHANNELS, n_hidden = 128, n_residual = 32, n_embeddings = 512, dim_embedding = 64, beta = utils.BETA)
    vqvae_dataset = Dataset(batch_size=utils.BATCH_SIZE, root_dir = utils.ADNI_ROOT_DIR, fraction=utils.VQVAE_FRACTION)
    if utils.VQVAE_RETRAIN :
        vqvae_trainer = TrainVQVAE(vqvae, vqvae_dataset, utils.VQVAE_LR, utils.VQVAE_WD, utils.VQVAE_EPOCHS, utils.VQVAE_SAVEPATH)
        vqvae_trainer.train()
        vqvae_trainer.plot(save=True)
        vqvae_trainer.save(utils.VQVAE_MODEL_PATH)
    else :
        vqvae.load_state_dict(torch.load(utils.VQVAE_MODEL_PATH))
    
    vqvae_tester = TestVQVAE(vqvae, vqvae_dataset, savepath=utils.VQVAE_SAVEPATH)
    vqvae_tester.reconstruct(path=utils.VQVAE_RECONSTRUCT_PATH, show=True)
    

    # gan = GAN(features = 128, latent_size = 128)
    # if utils.GAN_RETRAIN :
    #     gan_dataset = ModelDataset(vqvae, batch_size=utils.BATCH_SIZE, root_dir = utils.ADNI_ROOT_DIR, fraction=0.1)
    #     gan_trainer = TrainGAN(gan, gan_dataset, utils.GAN_LR, utils.GAN_WD, utils.GAN_EPOCHS, utils.GAN_SAVEPATH)
    #     gan_trainer.train()
    #     gan_trainer.plot(save=True)
    #     gan_trainer.save(utils.DISCRIMINATOR_PATH, utils.GENERATOR_PATH)
    # else :
    #     gan.discriminator.load_state_dict(torch.load(utils.DISCRIMINATOR_PATH))
    #     gan.generator.load_state_dict(torch.load(utils.GENERATOR_PATH))
    
    # generator = GenerateImages(vqvae, num_images=utils.NUM_IMAGES, device=device, savepath=utils.OUTPUT_PATH)
    # generator.generate()
    # generator.visualise()

    # noise = torch.randn(128, 128, 1, 1).to(device)

    # predictor = Predict(noise, utils.NUM_IMAGES, savepath=utils.OUTPUT_PATH, model=gan)
    # predictor.generate()
    # predictor.show_generated(save=True)