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

'''
Plot the training losses for both the generator and discriminator models
     generator_loss_file    Full path reference to the training loss file
     generator_loss_file    Full path reference to the training loss file
     
'''
def plot_losses(generator_loss_file, discriminator_loss_file):
    losses_generator = np.loadtxt(generator_loss_file)
    losses_discriminator = np.loadtxt(discriminator_loss_file)
    plt.plot(losses_generator, label="Generator loss")
    plt.plot(losses_discriminator, label="Discriminator loss")
    plt.xlabel("Batch Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("GAN Losses of Generator and Discriminiator")
    
    
def run():
    plot_losses(r"C:\Temp\losses_generator.csv", r"C:\Temp\losses_discriminator.csv")
    