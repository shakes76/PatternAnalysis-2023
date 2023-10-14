##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
from modules import VQVAE
import utils
from dataset import Dataset
from train import TrainVQVAE
from predict import Predict
from test import TestVQVAE
import matplotlib.pyplot as plt
import torch.nn.functional as F

if __name__ == '__main__':
    device = utils.DEVICE

    vqvae =  VQVAE(channels = utils.CHANNELS, 
                n_hidden = utils.VQVAE_HIDDEN, 
                n_residual = utils.VQVAE_RESIDUAL , 
                n_embeddings = 512, 
                dim_embedding = 64, 
                beta = utils.BETA
    )

    vqvae_dataset = Dataset(batch_size=utils.BATCH_SIZE, root_dir = utils.ADNI_ROOT_DIR, fraction=utils.FRACTION)

    if utils.VQVAE_RETRAIN :
        vqvae_trainer = TrainVQVAE(vqvae, vqvae_dataset, utils.VQVAE_LR, utils.VQVAE_WD, utils.VQVAE_EPOCHS, utils.VQVAE_SAVEPATH)
        vqvae_trainer.train()
        vqvae_trainer.plot(save=True)
        vqvae_trainer.save(utils.VQVAE_MODEL_PATH)
    else :
        vqvae.load_state_dict(torch.load(utils.VQVAE_RANGPUR_MODEL_PATH, map_location=utils.DEVICE)) # Change back to utils.VQVAE_MODEL_PATH
    
    if utils.VQVAE_TEST :
        vqvae_tester = TestVQVAE(vqvae, vqvae_dataset, savepath=utils.VQVAE_SAVEPATH)
        vqvae_tester.reconstruct(path=utils.VQVAE_RECONSTRUCT_PATH, show=True)
    
    if utils.VQVAE_PREDICT :
        codebook = vqvae.quantizer.embedding.weight.data
        print(codebook)
        generator = Predict(vqvae, num_images=utils.NUM_IMAGES, device=device, savepath=utils.OUTPUT_PATH, dataset=vqvae_dataset)
        generator.generate()
        # generator.generate()
        generator.visualise(show=False)
        generator.ssim()