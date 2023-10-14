##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
import matplotlib.pyplot as plt
import numpy as np

class Predict() :
    def __init__(self, input, n = 16, savepath = './', model = None, path = None) :
        self.savepath = savepath
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if model is None and path is not None :
            model = torch.load(path)
            self.model = model.to(self.device)
        else :
            self.model = model

        self.model.eval()
        self.output = None
        self.input = input
        self.n = n

    def generate(self) :
        with torch.no_grad() :
            output = self.model(self.input)
            self.output = output.cpu().numpy()
    
    def show_generated(self, save = True) :
        if self.output is None :
            self.generate()
            
        for i in range(self.n):
            plt.imshow(np.transpose(self.output[i], (1, 2, 0)))
            plt.axis('off')
            if save :
                plt.savefig(self.savepath + '_generated.png')
            else :
                plt.show()

class GenerateImages() :
    def __init__(self, model, num_images = 1, device='cpu', savepath = '') -> None:
        self.model = model
        self.n = num_images
        self.device = device
        self.savepath = savepath

        self.generated_images = None
        
    def generate_images(vqvae, num_images, image_size=(1, 64, 32, 32)):
        vqvae.eval()

        # Sample random indices from embedding space
        embedding_indices = torch.randint(
            high=vqvae.quantizer.n_embeddings,  # maximum embedding index
            size=(num_images, *image_size),     # size of the generated image batch
            device='cuda' if torch.cuda.is_available() else 'cpu' # device
        )

        # Convert embedding indices to one-hot encoded tensors
        embedding_one_hot = torch.nn.functional.one_hot(
            embedding_indices,
            num_classes=vqvae.quantizer.n_embeddings
        ).float()

        # Weight the one-hot tensor by the embedding weight to retrieve z_q
        z_q = torch.einsum('bchw,ce->bceh', embedding_one_hot, vqvae.quantizer.embedding.weight)

        # Use the decoder to generate images
        with torch.no_grad():  # ensure no gradients are computed
            generated_images = vqvae.decoder(z_q)

        return generated_images
    def generate(self):

        self.model.eval()
        random_embeddings = torch.randn(self.n, 64, 32, 32).to(self.device)
        print(f"Shape of random embeddings: {random_embeddings.shape}")

        with torch.no_grad():  # Ensure no gradients are calculated
            self.generated_images = self.model.decoder(random_embeddings)


    def visualise(self):
        """
        Visualize generated images.

        Parameters:
        - images (torch.Tensor): Tensor containing images to display.
        - self.n (int): Number of images to display.
        """
        
        rows = 4 
        cols = self.n // rows
        fig, axs = plt.subplots(rows, cols, figsize=(8, 4))
        axs = axs.ravel()
        axs = [axs] if self.n == 1 else axs

        

        for i, ax in enumerate(self.generated_images):
            axs[i].imshow(self.generated_images[i][0], cmap='gray')
            axs[i].axis('off')
        
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.savepath + f'images_new_{i}.png')
    
class SSIM():
    def __init__() :
        pass