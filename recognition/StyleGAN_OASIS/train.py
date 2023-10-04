# -*- coding: utf-8 -*-
"""
File: train.py

This is a placeholder file

@author: Peter Beardsley
"""

import numpy as np
import torch
from torch import optim
from dataset import createDataLoader, getDataSize
from modules import AlphaScheduler, Generator, Discriminator


"""
Implementation of gradient penalty. Full credit of this funtion goes to 
https://blog.paperspace.com/implementation-stylegan-from-scratch/ which get 
their code from the gradient_penalty function for WGAN-GP loss.

Note: This code is very specific and crafting it independently is beyond the
scope of this course. So with full references and acknowledgement I am using
it as is, unaltered.
"""
def gradient_penalty(discriminator, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate discriminator scores
    mixed_scores = discriminator(interpolated_images)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty




"""
Run the training loop for the StyleGAN. This will simulanteously train the
Generator and Discriminator as they adversarially work together.

    cfg     Dictionary of training options (see README.md for details)
    device  Either CPU or the CUDA device
"""
def train_stylegan(cfg, device):
    data_size = getDataSize(cfg['image_folder'])
    alphaSched = AlphaScheduler(cfg['fade_epochs'], cfg['batch_sizes'], data_size)
    generator = Generator(cfg['z_size'], cfg['w_size'], cfg['channels'], cfg['rgb_ch'], alphaSched, is_progressive=False).to(device)
    discriminator = Discriminator(cfg['channels'], cfg['rgb_ch'], alphaSched, is_progressive=False).to(device)
    
    
    # initialize optimizers
    optimiser_generator = optim.Adam([{"params": [param for name, param in generator.named_parameters() if "mappingNetwork" not in name]},
                            {"params": generator.mappingNetwork.parameters(), "lr": cfg['mapping_lr']}], lr=cfg['lr'], betas=(0.0, 0.99))
    optimiser_discriminator = optim.Adam(discriminator.parameters(), lr=cfg['lr'], betas=(0.0, 0.99))
    
    generator.train()
    discriminator.train()
    
    # start at step that corresponds to img size that we set in config
    for layer_epochs in cfg['fade_epochs'][1:]:
        loader = createDataLoader(cfg['image_folder'], 4 * 2 ** alphaSched.depth, cfg['batch_sizes'][alphaSched.depth-1])  
    
        for epoch in range(layer_epochs):
            for real, _ in loader:
                real = real.to(device)
                       
                z = torch.randn(real.shape[0], cfg['z_size']).to(device)                    # Generator the latent z
        
                fake = generator(z)                                                         # Generator the fake images from z
                discriminator_real = discriminator(real)                                    # Discriminate the real images
                discriminator_fake = discriminator(fake.detach())                           # Discriminate the fake images
                
                """
                The discriminator loss calculation is not my work. The technique
                was obtained from https://blog.paperspace.com/implementation-stylegan-from-scratch/
                with full credit to auther Abd Elilah TAUIL. In my observation, 
                this code needed to run exactly as written or the StyleGAN fails 
                to converge. To get this working, I had to remove the Sigmoid()
                function from the discriminator.
                """
                gp = gradient_penalty(discriminator, real, fake, device=device)
                loss_discriminator = (
                    -(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
                    + cfg['lambda'] * gp
                    + (0.001 * torch.mean(discriminator_real ** 2))
                )
        
                # 
                discriminator.zero_grad()
                loss_discriminator.backward()
                optimiser_discriminator.step()
        
                generator_fake = discriminator(fake)
                loss_generator = -torch.mean(generator_fake)
        
                generator.zero_grad()
                loss_generator.backward()
                optimiser_generator.step()
                
                alphaSched.stepAlpha()
                print('Epoch:', epoch+1,"/",layer_epochs, "   Depth:", alphaSched.depth, "   Alpha:", alphaSched.alpha)

    
        torch.save(generator.state_dict(), cfg['generator_model_file'].format(alphaSched.depth))
        alphaSched.stepDepth()


"""
Train a styleGAN on the OASIS brain dataset using a predefined configuration
"""
def train_stylegan_oasis():
    cfg = {'epochs':20, 'batch_sizes':[256,128,128,64,16,8,8], 'channels':[512, 512,256,128,64,32,16,4], 'rgb_ch':3
            , 'fade_epochs': np.array([1, 1, 1, 1, 1, 1, 1])
            , 'depth': 6
            , 'image_folder': r'C:\Temp\keras_png_slices_data\keras_png_slices_data'
            , 'generator_model_file': 'C:\Temp\stylegan_depth_{0}.pth'
            , 'discriminator_model_file': 'C:\Temp\gan_dis_depth_{0}.pth'
            , 'discriminator_loss_file': r'C:\Temp\losses_discriminator.csv'
            , 'generator_loss_file': r'C:\Temp\losses_generator.csv'
            , 'lr':0.001, 'mapping_lr':0.00001, 'lambda':10, 'beta1': 0.0, 'beta2': 0.999, 'z_size':256, 'w_size':256, 'img_size': 256}
        

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_stylegan(cfg, device)