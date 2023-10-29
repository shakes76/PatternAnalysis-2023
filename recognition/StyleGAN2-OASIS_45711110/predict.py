#example usage of the trained model

import torch
import os
import train

import matplotlib.pyplot as plt
from torchvision.utils import save_image

'''
Generate Imagees using the generator.
Images are saved in separate epoch folder with 100 images each

Epoch intervals is sent as parameter while the number of imgs and path is hard coded below.
'''
def generate_examples(gen, epoch, n=100):
    
    for i in range(n):
        with torch.no_grad():
            w     = train.get_w(1)
            noise = train.get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

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