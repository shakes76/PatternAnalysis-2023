# -*- coding: utf-8 -*-
"""
File: predict.py

Purpose: Tools to generate content from the StyleGAN at various synthesis depths
         and epochs, as well as visualise training losses of both the generator
         and the discriminator.

@author: Peter Beardsley
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from modules import Generator, AlphaScheduler

'''
Plot the training losses for both the generator and discriminator models
     loss_file    Full path reference to the training loss file. Data in
                  a NxF structure, N = number of batch iterations,
                  F = [Weighted Gradient Penalty
                      Wasserstein Distance
                      Gradient Clipping
                      Total Discriminator Loss
                      Total Discriminator Loss]
'''
def plot_losses(loss_file):
    losses = np.loadtxt(loss_file)

    plt.plot(losses[:,4], label="Total Discriminator Loss")    
    plt.plot(losses[:,3], label="Total Discriminator Loss")
    plt.plot(losses[:,0], label="Weighted Gradient Penalty")
    plt.plot(losses[:,1], label="Wasserstein Distance")
    plt.plot(losses[:,2], label="Gradient Clipping")

    plt.xlabel("Batch Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Losses of Generator and Discriminiator")
    
"""
TODO: x
"""
def plot_generation(gen, n):
    gen.eval()
    z = torch.randn(n, gen.z_size)
    gen_images = gen(z)
    plt.figure(figsize=(20,20))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(gen_images[:n], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.title(f"Depth: {gen.alphaSched.depth}")
    plt.show()
    
    

def run():
    z_size = 256
    w_size = 256
    rgb_ch = 3
    channels = [512, 512,256,128,64,32,16,4]
    alphaSched = AlphaScheduler([1], [1], 1)
    alphaSched.depth=6
    generator = Generator(z_size, w_size, channels, rgb_ch, alphaSched, is_progressive=False)
    generator.load_state_dict(torch.load("C:\\Temp\\stylegan_depth_6.pth"))
    plot_generation(generator, 64, )
    
    plot_losses(r"C:\Temp\losses_generator.csv", r"C:\Temp\losses_discriminator.csv")



run()
