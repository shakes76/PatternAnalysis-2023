"""
Created on Sunday Oct 8 2:22:00 2023

This script is for Setting up the code that will be used for training the DCGAN model.

@author: Gabriel Russell
@ID: s4640776

"""
"""
Class that contains parameters and training of the DCGAN
"""
from modules import *
from dataset import *
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt



"""
This Class handles the training part of the DCGAN. 
It takes in a data loader formed from the encoding indices 
of the trained VQVAE model.
"""
#Referenced from
#https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/2.%20DCGAN/train.py
class TrainDCGAN():
    def __init__(self, train_loader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = Parameters()
        self.Discriminator = Discriminator().to(self.device)
        self.Generator = Generator().to(self.device)
        initialize_weights(self.Discriminator)
        initialize_weights(self.Generator)

        self.epochs = 50
        self.D_optim = optim.Adam(self.Discriminator.parameters(), lr = self.params.gan_lr, betas = (0.5, 0.999))
        self.G_optim = optim.Adam(self.Generator.parameters(), lr = self.params.gan_lr, betas = (0.5, 0.999))
        self.criterion = nn.BCELoss()

        self.train_loader = train_loader

    def train(self):
        discriminator_loss = []
        generator_loss = []
        self.Generator.train()
        self.Discriminator.train()
        for epoch in range(self.epochs):
            print("DCGAN Training Epoch: ", epoch + 1)
            for i in enumerate(self.train_loader):
                num, batch = i
                real_img = batch.to(self.device)
                batch_size = real_img.shape[0]

                #Generate fake image to pass through to model
                rand_noise = torch.randn(batch_size, 100,1,1).to(self.device)
                fake_img = self.Generator(rand_noise)


                #Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
                D_real = self.Discriminator(real_img).reshape(-1)
                D_real_loss = self.criterion(D_real, torch.ones_like(D_real))
                D_fake = self.Discriminator(fake_img.detach()).reshape(-1)
                D_fake_loss = self.criterion(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_fake_loss + D_real_loss)/2
                self.Discriminator.zero_grad()
                D_loss.backward()
                self.D_optim.step()

                #Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
                out = self.Discriminator(fake_img).reshape(-1)
                G_loss = self.criterion(out, torch.ones_like(out))
                self.Generator.zero_grad()
                G_loss.backward()
                self.G_optim.step()

                discriminator_loss.append(D_loss.item())
                generator_loss.append(G_loss.item())

                if num % 50 == 0:
                    print(
                f"Epoch [{epoch}/{self.epochs}] Batch {num}/{len(self.train_loader)} \
                  Loss D: {D_loss:.4f}, loss G: {G_loss:.4f}"
            )
            #Save images to look at progress
            if epoch % 2 == 0:
                save_image(fake_img.data[:25], f"gan_images/epoch_{epoch}.png", nrow=5, normalize=True)
        current_dir = os.getcwd()
        D_model_path = current_dir + "/Models/Discriminator.pth"
        G_model_path = current_dir + "/Models/Generator.pth"
        torch.save(self.Discriminator, D_model_path)
        torch.save(self.Generator, G_model_path)

        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(discriminator_loss, label = "Discriminator")
        ax.plot(generator_loss, label = "Generator")
        ax.set_ylabel('Loss')
        ax.set_title('DCGAN losses during training')
        ax.set_xlabel('Iterations')
        ax.legend()
        plt.savefig("Output_files/Discriminator and Generator Losses.png")

     
