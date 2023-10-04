# -*- coding: utf-8 -*-
"""
File: train.py

This is a placeholder file

@author: Peter Beardsley
"""

import numpy as np
import torch
from torch import optim, nn
from dataset import createDataLoader, getDataSize
from model import AlphaScheduler, Generator, Discriminator

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
    gen_size=cfg['gen_size']
    
    # initialize optimizers
    optimizer_generator = optim.Adam([{"params": [param for name, param in generator.named_parameters() if "mappingNetwork" not in name]},
                            {"params": generator.mappingNetwork.parameters(), "lr": cfg['mapping_lr']}], lr=cfg['lr'], betas=(0.0, 0.99))
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=cfg['lr'], betas=(0.0, 0.99))
    
    loss_fn = nn.BCELoss()
    
    real_label = 1.0
    fake_label = 0.0
    
    generator.train()
    discriminator.train()
    
    Dloss = []
    Gloss = []
    
    # start at step that corresponds to img size that we set in config
    for layer_epochs in cfg['fade_epochs'][1:]:
        loader = createDataLoader(cfg['image_folder'], 4 * 2 ** alphaSched.depth, cfg['batch_sizes'][alphaSched.depth-1])  
    
        for epoch in range(layer_epochs):
            for real, _ in loader:
                real = real.to(device)
        
                discriminator.zero_grad()                                                                 # Clear gradients ready for training
                data_real = real.to(device)                                                               # Move batch to GPU
                label_real = torch.full((len(data_real),), real_label, dtype=torch.float, device=device)  # Label training as real
                output = discriminator(data_real).view(-1)                                                # Classify the real images
                loss_real = loss_fn(output, label_real)                                                   # Get the loss of the real images
                loss_real.backward()                                                                      # Get the gradient from the real images
                gen_data = torch.randn(len(data_real), 1, 1, gen_size, device=device)                     # Generator random input (seed)
                data_fake = generator(gen_data)                                                           # Generator fake images from the seed
                label_fake = torch.full((len(data_fake),), fake_label, dtype=torch.float, device=device)  # Label training as fake
                output = discriminator(data_fake.detach()).view(-1)                                       # Classify the fake images
                loss_fake = loss_fn(output, label_fake)                                                   # Get the loss of the fake images
                loss_fake.backward()                                                                      # Get the gradient from the fake images
                loss = 0.1*loss_real + 0.9*loss_fake                                                      # Combine losses of real and fake
                optimizer_discriminator.step()                                                            # Update the Discriminator from the two parts
                Dloss.append(loss.detach().cpu())

                # Train the Generator
                generator.zero_grad()                                                                     # Clear the gradients ready for training
                output = discriminator(data_fake).view(-1)                                                # Discriminator was updated, so re-classifiy fakes
                loss = loss_fn(output, label_real)                                                        # Get the loss of the fakes using real labels*
                loss.backward()                   
                optimizer_generator.step()                                                                # Update the Generator
                Gloss.append(loss.detach().cpu())
                
                alphaSched.stepAlpha()
                print('Epoch:', epoch, "   Depth:", alphaSched.depth, "   Alpha:", alphaSched.alpha)

    
        torch.save(generator.state_dict(), f'./stylegan_depth_{alphaSched.depth}.pth')


"""
Train a styleGAN on the OASIS brain dataset using a predefined configuration
"""
def train_stylegan_oasis():
    cfg = {'epochs':20, 'batch_sizes':[256,128,128,64,16,8,8], 'channels':[512, 512,256,128,64,32,16,4], 'rgb_ch':3
            , 'fade_epochs': np.array([1, 1, 1, 1, 1, 1, 1])
            , 'depth': 6
            , 'image_folder': r'C:\Temp\keras_png_slices_data\keras_png_slices_data'
            , 'generator_model_file': 'C:\Temp\gan_gen_depth_{0}.pth'
            , 'discriminator_model_file': 'C:\Temp\gan_dis_depth_{0}.pth'
            , 'discriminator_loss_file': r'C:\Temp\losses_discriminator.csv'
            , 'generator_loss_file': r'C:\Temp\losses_generator.csv'
            , 'lr':0.001, 'mapping_lr':0.00001, 'lambda':10, 'beta1': 0.0, 'beta2': 0.999, 'z_size':256, 'w_size':256, 'img_size': 256}
        

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_stylegan(cfg, device)