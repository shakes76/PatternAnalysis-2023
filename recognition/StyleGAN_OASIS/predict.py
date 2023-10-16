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
from dataset import createDataLoader

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
    plt.plot(losses[:,3], label="Total Generator Loss")
    plt.xlabel("Batch Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Losses of Generator and Discriminiator")
    plt.show()
    
    plt.plot(losses[:,0], label="Weighted Gradient Penalty")
    plt.plot(losses[:,1], label="Wasserstein Distance")
    plt.plot(losses[:,2], label="Gradient Clipping")   
    plt.xlabel("Batch Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Decomposition of GAN Losses of Discriminator")
"""
Plot n images from synthetically generated images 
    generator       A training generator class with weights preloaded
    n               Number of images to dispay
"""
def plot_synthetic(generator, n, subtitle):
    generator.eval()
    z = torch.randn(n, 256)
    images = generator(z)
    plt.figure(figsize=(16,16))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(images[:n], nrow=int(np.sqrt(n)), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.title("Synthetically Generated Images - " + subtitle)
    plt.show()
    

"""
Plot n real images 
    dataloader      A dataloader with a batch size of at least n
    n               Number of images to dispay
"""
def plot_real(dataloader, n):
    images = next(iter(dataloader))[0]
    plt.figure(figsize=(16,16))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(images[:n], nrow=int(np.sqrt(n)), padding=2, normalize=True).cpu(),(1,2,0)))
    plt.title("Real Images - 256x256")
    plt.show()   
    

