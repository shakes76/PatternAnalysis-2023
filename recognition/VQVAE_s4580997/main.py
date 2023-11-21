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

if __name__ == '__main__':
    # Models
    vqvae = VQVAE(channels = utils.CHANNELS, 
                n_hidden = utils.VQVAE_HIDDEN, 
                n_residual = utils.VQVAE_RESIDUAL , 
                n_embeddings = utils.VQVAE_EMBEDDINGS, 
                dim_embedding = utils.VQVAE_EMBEDDING_DIM,
                beta = utils.BETA
    )
    vqvae = vqvae.to(utils.DEVICE)

    gan = GAN(utils.CHANNELS, utils.NOISE, utils.GAN_IMG_SIZE)
    gan = gan.to(utils.DEVICE)

    # Core dataset
    adni_dataset = Dataset(batch_size=utils.BATCH_SIZE, root_dir = utils.ADNI_ROOT_DIR, fraction=utils.FRACTION)

    # Train VQVAE
    if utils.VQVAE_RETRAIN :
        vqvae_trainer = TrainVQVAE(vqvae, adni_dataset, utils.VQVAE_LR, utils.VQVAE_WD, utils.VQVAE_EPOCHS, utils.VQVAE_SAVEPATH)
        vqvae_trainer.train()
        vqvae_trainer.validate()
        vqvae_trainer.plot(save=True)
        vqvae_trainer.save(utils.VQVAE_MODEL_PATH)
    else :
        vqvae.load_state_dict(torch.load(utils.VQVAE_MODEL_PATH, map_location=utils.DEVICE)) 
    
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
    
    # Run predict
    if utils.PREDICT :
        predict = Predict(vqvae, gan, adni_dataset, utils.DEVICE, savepath=utils.OUTPUT_PATH, img_size=64)
        predict.generate_gan()
        predict.generate_vqvae()
        predict.ssim()
