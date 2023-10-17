##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
from modules import VQVAE, GAN, PixelCNN
import utils
from dataset import Dataset, ModelDataset
from train import TrainVQVAE, TrainGAN, TrainPixelCNN
from predict import Predict
from test import TestVQVAE
import matplotlib.pyplot as plt
import torch.nn.functional as F
import other

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
    # pixelcnn = PixelCNN(utils.VQVAE_HIDDEN, utils.GAN_LATENT_DIM)

    # Core dataset
    adni_dataset = Dataset(batch_size=utils.BATCH_SIZE, root_dir = utils.ADNI_ROOT_DIR, fraction=utils.FRACTION)

    # Train VQVAE
    if utils.VQVAE_RETRAIN :
        vqvae_trainer = TrainVQVAE(vqvae, adni_dataset, utils.VQVAE_LR, utils.VQVAE_WD, utils.VQVAE_EPOCHS, utils.VQVAE_SAVEPATH)
        vqvae_trainer.train()
        vqvae_trainer.plot(save=True)
        vqvae_trainer.save(utils.VQVAE_MODEL_PATH)
    else :
        # vqvae.load_state_dict(torch.load(utils.VQVAE_MODEL_PATH, map_location=utils.DEVICE)) # Change back to utils.VQVAE_MODEL_PATH
        vqvae.load_state_dict(torch.load(utils.VQVAE_RANGPUR_MODEL_PATH, map_location=utils.DEVICE)) # Change back to utils.VQVAE_MODEL_PATH
    
    # if utils.PIXELCNN_RETRAIN :
    #     pixel_trainer = TrainPixelCNN(vqvae, pixelcnn, adni_dataset, utils.VQVAE_LR, utils.VQVAE_WD, utils.GAN_EPOCHS, utils.PIXEL_SAVEPATH)
    #     pixel_trainer.train()
    #     pixel_trainer.plot(save = True)
    #     pixel_trainer.save(utils.PIXEL_MODEL_PATH)
    # else :
    #     # pixelcnn.load_state_dict(torch.load(utils.PIXEL_MODEL_PATH, map_location=utils.DEVICE))
    #     pixelcnn.load_state_dict(torch.load(utils.PIXEL_RANGPUR_MODEL_PATH, map_location=utils.DEVICE))

    # Train GAN prior
    if utils.GAN_RETRAIN :
        gan_dataset = ModelDataset(vqvae, batch_size=utils.BATCH_SIZE, root_dir = utils.ADNI_ROOT_DIR, fraction=utils.FRACTION)
        gan_trainer = TrainGAN(gan, gan_dataset, utils.GAN_LR, utils.GAN_WD, utils.GAN_EPOCHS, utils.GAN_SAVEPATH)
        gan_trainer.train()
        gan_trainer.plot(save=True)
        gan_trainer.save(utils.DISCRIMINATOR_MODEL_PATH, utils.GENERATOR_MODEL_PATH)
    else :
        # gan.discriminator.load_state_dict(torch.load(utils.DISCRIMINATOR_MODEL_PATH, map_location=utils.DEVICE))
        # gan.generator.load_state_dict(torch.load(utils.GENERATOR_MODEL_PATH, map_location=utils.DEVICE))
        gan.discriminator.load_state_dict(torch.load(utils.DISCRIMINATOR_RANGPUR_MODEL_PATH, map_location=utils.DEVICE))
        gan.generator.load_state_dict(torch.load(utils.GENERATOR_RANGPUR_MODEL_PATH, map_location=utils.DEVICE))

    
    # Run test
    if utils.VQVAE_TEST :
        vqvae_tester = TestVQVAE(vqvae, adni_dataset, savepath=utils.VQVAE_SAVEPATH, device=utils.DEVICE)
        vqvae_tester.reconstruct(path=utils.VQVAE_RECONSTRUCT_PATH, show=True)
    
    # Run predict
    if utils.VQVAE_PREDICT :
        predict = Predict(vqvae, gan, adni_dataset, utils.DEVICE, savepath=utils.OUTPUT_PATH, img_size=64)
        predict.generate_gan(1)
        predict.generate_vqvae(1)
        # predict.ssim('gan')
        # predict.ssim('vqvae')
        # predict.generate_pixelcnn_vqvae(pixelcnn)
        # predict.pixel(pixelcnn, 32)
        # predict.gan_generated_images(gan.generator, 128, utils.DEVICE)

        generated_images = other.gan_generated_images(gan, utils.DEVICE)

        #Function to save visualisation of generated code indice
        code_indice = other.gan_create_codebook_indice(generated_images)

        #Function for decoding the generated outputs and save as final reconstruction
        decoded = other.gan_reconstruct(vqvae, code_indice)

        #Calculate the average and max SSIM against test data set and print to terminal
        other.SSIM(decoded)