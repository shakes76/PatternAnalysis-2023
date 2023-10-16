# -*- coding: utf-8 -*
"""
File: test_driver_predict.py

Purpose: A driver script that runs inference on a trained StyleGAN of the OASIS Brain dataset

@author: Peter Beardsley
"""

from predict import plot_synthetic, plot_real, plot_losses
from modules import Generator, AlphaScheduler
from dataset import createDataLoader
import numpy as np
import torch


"""
Run the predict analysis
This will plot synthetic and real images, as well as training losses
"""
# This is the same config from training.
cfg = {'batch_sizes':[256,128,128,64,16,8,8], 'channels':[512, 512,256,128,64,32,16,8,4], 'rgb_ch':3
        , 'fade_epochs': np.array([30, 30, 30, 30, 30, 30, 30, 30])
        , 'image_folder': r'C:\Temp\keras_png_slices_data\keras_png_slices_data'
        , 'generator_model_file': 'C:\Temp\stylegan_depth_{0}.pth'
        , 'losses_file': 'C:\Temp\losses.csv'
        , 'lr':0.001, 'mapping_lr':1e-5,'lambda':10, 'beta1': 0.0, 'beta2': 0.999, 'z_size':256, 'w_size':256, 'img_size': 256}
    

# The AlphaScheduler is linked to the Generator, so configure, set to depth 6 (256x256) and load the saved model
alphaSched = AlphaScheduler(cfg['fade_epochs'], cfg['batch_sizes'], 1e10)
alphaSched.depth=6
generator = Generator(cfg['z_size'], cfg['w_size'], cfg['channels'], cfg['rgb_ch'], alphaSched)
generator.load_state_dict(torch.load(cfg['generator_model_file'].format(alphaSched.depth)))

# Plot the generated synthetic images at input size of 256x256
plot_synthetic(generator, n=16, subtitle='256x256')

alphaSched.depth=7
generator.load_state_dict(torch.load(cfg['generator_model_file'].format(alphaSched.depth)))

# Plot the generated synthetic images at super size of 512x512
plot_synthetic(generator, n=4, subtitle='512x512')

# Load and plot some real images
dataloader = createDataLoader(cfg['image_folder'], 256, 256)
plot_real(dataloader, n=16)

# Load and plot the training losses
plot_losses(cfg['losses_file'])

