# -*- coding: utf-8 -*-
"""
File: train.py

Purpose: Run the training loop for a StyleGAN that implements special loss calculations
         specific to StyleGANs, as well as alpha blending for Progressive GANs

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
scope of this course. So with full reference and acknowledgement I am using
it as is, unaltered. Credit to Abd Elilah TAUIL.
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
    generator = Generator(cfg['z_size'], cfg['w_size'], cfg['channels'], cfg['rgb_ch'], alphaSched).to(device)
    discriminator = Discriminator(cfg['channels'], cfg['rgb_ch'], alphaSched).to(device)
    
    
    # initialize optimizers
    # Some StyleGAN implementations scale the Mapping Network weights by 0.01 to reduce the learning rate. I've
    # come across this alternative approach a few times and prefer it for its explicit transparency. It specifically
    # allocates a different learning rate for the mappingNetwork parameter.
    optimiser_generator = optim.Adam([{"params": [param for name, param in generator.named_parameters() if "mappingNetwork" not in name]},
                            {"params": generator.mappingNetwork.parameters(), "lr": cfg['mapping_lr']}], lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
    optimiser_discriminator = optim.Adam(discriminator.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
    
    # Set to training mode
    generator.train()
    discriminator.train()
    
    # Skip the 4x4 training and iterate from 8x8 onwards
    for layer_epochs in cfg['fade_epochs'][1:]:
        loader = createDataLoader(cfg['image_folder'], 4 * 2 ** alphaSched.depth, cfg['batch_sizes'][alphaSched.depth])  
    
        # Train for the number of epochs specified in the fade
        for epoch in range(layer_epochs):
            for real, _ in loader:
                real = real.to(device)
                       
                z = torch.randn(real.shape[0], cfg['z_size']).to(device)                    # Generator the latent z
        
                fake = generator(z)                                                         # Generator the fake images from z
                discriminator_real = discriminator(real)                                    # Discriminate the real images
                discriminator_fake = discriminator(fake.detach())                           # Discriminate the fake images
                
                """
                The discriminator loss calculation is not my work. I have adjusted
                the code to improve its interperability. The technique was
                obtained from https://blog.paperspace.com/implementation-stylegan-from-scratch/
                with full credit to auther Abd Elilah TAUIL. In my observation, 
                this code needed to run exactly as written or the StyleGAN fails 
                to converge and suffers from mode collapse. To get this working, 
                I had to remove the Sigmoid() function from the discriminator.
                See README.md for detailed explainations.
                """
                gp = gradient_penalty(discriminator, real, fake, device=device)
                weighted_gradient_penalty = cfg['lambda'] * gp
                wasserstein_distance = -(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
                gradient_clipping = (0.001 * torch.mean(discriminator_real ** 2))
                loss_discriminator = wasserstein_distance + weighted_gradient_penalty + gradient_clipping
        
                # Back propogate the discriminator losses
                discriminator.zero_grad()
                loss_discriminator.backward()
                optimiser_discriminator.step()
        
                # Calculate the Generator loss
                generator_fake = discriminator(fake)
                loss_generator = -torch.mean(generator_fake)
        
                # Back propogate the generator loss
                generator.zero_grad()
                loss_generator.backward()
                optimiser_generator.step()
                
                # Write out batch losses for post-training analysis
                with open(cfg['losses_file'], "a") as f:
                    np.savetxt(f, [[weighted_gradient_penalty.detach().cpu()
                                   , wasserstein_distance.detach().cpu()
                                   , gradient_clipping.detach().cpu()
                                   , loss_discriminator.detach().cpu()
                                   , loss_generator.detach().cpu()]])
                               
                alphaSched.stepAlpha()
                print('Epoch:', epoch+1,"/",layer_epochs, "   Depth:", alphaSched.depth, "   Alpha:", alphaSched.alpha)

    
        alphaSched.stepDepth()
        torch.save(generator.state_dict(), cfg['generator_model_file'].format(alphaSched.depth))
        

"""
Train a styleGAN on the OASIS brain dataset using a predefined configuration
"""
def train_stylegan_oasis(is_rangpur=True):
    cfg_win = {'batch_sizes':[256,128,128,64,16,8,8], 'channels':[512, 512,256,128,64,32,16,8,4], 'rgb_ch':3
            , 'fade_epochs': np.array([2, 2, 2, 2, 2, 2, 2, 2])
            , 'image_folder': r'C:\Temp\keras_png_slices_data\keras_png_slices_data'
            , 'generator_model_file': 'C:\Temp\stylegan_depth_{0}.pth'
            , 'losses_file': 'losses.csv'
            , 'lr':0.001, 'mapping_lr':1e-5,'lambda':10, 'beta1': 0.0, 'beta2': 0.999, 'z_size':256, 'w_size':256, 'img_size': 256}
    
    cfg_rangpur = {'batch_sizes':[256,128,128,64,16,8,8], 'channels':[512, 512,256,128,64,32,16,8,4], 'rgb_ch':3
            , 'fade_epochs': np.array([30, 30, 30, 30, 30, 30, 30, 30])
            , 'image_folder': '/home/groups/comp3710/OASIS'
            , 'generator_model_file': 'stylegan_depth_{0}.pth'
            , 'losses_file': 'losses.csv'
            , 'lr':0.001, 'mapping_lr':1e-5, 'lambda':10, 'beta1': 0.0, 'beta2': 0.999, 'z_size':256, 'w_size':256, 'img_size': 256}

    if is_rangpur:
        cfg = cfg_rangpur
    else: 
        cfg = cfg_win

        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_stylegan(cfg, device)
