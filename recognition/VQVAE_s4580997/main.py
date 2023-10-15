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
from predict import Predict
from test import TestVQVAE
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == '__main__':
    # Models
    vqvae =  VQVAE(channels = utils.CHANNELS, 
                n_hidden = utils.VQVAE_HIDDEN, 
                n_residual = utils.VQVAE_RESIDUAL , 
                n_embeddings = 512, 
                dim_embedding = 64, 
                beta = utils.BETA
    )
    gan = GAN()

    # Core dataset
    vqvae_dataset = Dataset(batch_size=utils.BATCH_SIZE, root_dir = utils.ADNI_ROOT_DIR, fraction=utils.FRACTION)

    # Train VQVAE
    if utils.VQVAE_RETRAIN :
        vqvae_trainer = TrainVQVAE(vqvae, vqvae_dataset, utils.VQVAE_LR, utils.VQVAE_WD, utils.VQVAE_EPOCHS, utils.VQVAE_SAVEPATH)
        vqvae_trainer.train()
        vqvae_trainer.plot(save=True)
        vqvae_trainer.save(utils.VQVAE_MODEL_PATH)
    else :
        vqvae.load_state_dict(torch.load(utils.VQVAE_RANGPUR_MODEL_PATH, map_location=utils.DEVICE)) # Change back to utils.VQVAE_MODEL_PATH
    
    # Train GAN prior
    if utils.GAN_RETRAIN :
        gan_dataset = ModelDataset(vqvae, batch_size=utils.BATCH_SIZE, root_dir = utils.ADNI_ROOT_DIR, fraction=utils.FRACTION)
        gan_trainer = TrainGAN(gan, gan_dataset, utils.GAN_LR, utils.GAN_WD, utils.GAN_EPOCHS, utils.GAN_SAVEPATH)
        gan_trainer.train()
        gan_trainer.plot(save=True)
        gan_trainer.save(utils.DISCRIMINATOR_MODEL_PATH, utils.GENERATOR_MODEL_PATH)
    else :
        gan.discriminator.load_state_dict(torch.load(utils.DISCRIMINATOR_MODEL_PATH, map_location=utils.DEVICE))
        gan.generator.load_state_dict(torch.load(utils.GENERATOR_MODEL_PATH, map_location=utils.DEVICE))

    # Run test
    if utils.VQVAE_TEST :
        vqvae_tester = TestVQVAE(vqvae, vqvae_dataset, savepath=utils.VQVAE_SAVEPATH)
        vqvae_tester.reconstruct(path=utils.VQVAE_RECONSTRUCT_PATH, show=True)
    
    # Run predict
    if utils.VQVAE_PREDICT :
        generator = Predict(vqvae, gan, vqvae_dataset, device=utils.DEVICE, savepath=utils.OUTPUT_PATH, img_size=(utils.VQVAE_RESIDUAL, utils.VQVAE_RESIDUAL))
        generator.generate_gan()
        generator.generate_vqvae()
        generator.ssim()