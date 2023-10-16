"""
Created on Sunday Oct 8 2:22:00 2023

This script is for Setting up the code that will be used for training the DCGAN model.

@author: Gabriel Russell
@ID: s4640776

"""
from modules import *
from dataset import *

#Referenced from
#https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/2.%20DCGAN/train.py
"""
This Class handles the training part of the DCGAN. 
It takes in a data loader formed from the encoding indices 
of the trained VQVAE model.
"""
class TrainDCGAN():
    def __init__(self, train_loader):
        """
        Initialises attributes for DCGAN training process

        Args: 
            train_loader (Dataloader): Training Dataset created from encoded images

        Returns:
            None
        """
        self.params = Parameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Create Models
        self.Discriminator = Discriminator().to(self.device)
        self.Generator = Generator().to(self.device)
        #Initialise weights for models
        initialize_weights(self.Discriminator)
        initialize_weights(self.Generator)
        #Create optimizers for each model
        self.D_optim = optim.Adam(self.Discriminator.parameters(), lr = self.params.gan_lr, betas = (0.5, 0.999))
        self.G_optim = optim.Adam(self.Generator.parameters(), lr = self.params.gan_lr, betas = (0.5, 0.999))
        self.criterion = nn.BCELoss()
        self.train_loader = train_loader
        self.epochs = 20

    def train(self):
        """
        Training function for DCGAN.
        Also saves models after training, produces loss
        plots for each network.

        Args: 
            None

        Returns:
            None
        """
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

                if num % 150 == 0:
                    print(
                f"Epoch [{epoch}/{self.epochs}] Batch {num}/{len(self.train_loader)} \
                  Loss D: {D_loss:.4f}, loss G: {G_loss:.4f}"
            )
            #Save images to look at progress during testing
            if epoch % 2 == 0:
                save_image(fake_img.data[:25], f"gan_images/epoch_{epoch}.png", nrow=5, normalize=True)
        
        #Save models to Model folder
        current_dir = os.getcwd()
        D_model_path = current_dir + "/Models/Discriminator.pth"
        G_model_path = current_dir + "/Models/Generator.pth"
        torch.save(self.Discriminator, D_model_path)
        torch.save(self.Generator, G_model_path)

        #Save Loss plot of Discriminator and Generator
        f = plt.figure(figsize=(8,8))
        ax = f.add_subplot(1,1,1)
        ax.plot(discriminator_loss, label = "Discriminator")
        ax.plot(generator_loss, label = "Generator")
        ax.set_ylabel('Loss')
        ax.set_title('DCGAN losses during training')
        ax.set_xlabel('Iterations')
        ax.legend()
        plt.savefig("Output_files/Discriminator and Generator Losses.png")

     
