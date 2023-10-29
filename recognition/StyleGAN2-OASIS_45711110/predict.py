#example usage of the trained model

import torch
import os
import train

import matplotlib.pyplot as plt
from torchvision.utils import save_image

'''
Plot a 10x5 graph of the Generator and Discriminator loss during training over iteration
'''
def plot_loss(G_Loss, D_Loss):
    plt.figure(figsize=(10,5))
    plt.title("Generator Loss During Training")
    plt.plot(G_Loss, label="G", color="blue")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('gen_loss.png')

    plt.figure(figsize=(10,5))
    plt.title("Discriminator Loss During Training")
    plt.plot(D_Loss, label="D", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('disc_loss.png')