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
        
    def generate(self):

        self.model.eval()
        random_embeddings = torch.randn(self.n, 64, 32, 32).to(self.device)
        print(f"Shape of random embeddings: {random_embeddings.shape}")

        try:
            with torch.no_grad():  # Ensure no gradients are calculated
                self.generated_images = self.model.decoder(random_embeddings)
        except RuntimeError as e:
            print(f"Runtime error encountered: {str(e)}")
            raise e

    def visualise(self):
        """
        Visualize generated images.

        Parameters:
        - images (torch.Tensor): Tensor containing images to display.
        - self.n (int): Number of images to display.
        """
        
        # Move images to CPU and convert them to numpy
        images_np = self.generated_images.cpu().detach().numpy()
        
        # Choose the first 'self.n' to display
        images_to_display = images_np[:self.n]

        # Assume image shape [self.n, num_channels, height, width]
        _, num_channels, height, width = images_to_display.shape
        
        fig, axs = plt.subplots(1, self.n, figsize=(8, 8))
        axs = [axs] if self.n == 1 else axs

        for i, ax in enumerate(axs):
            if num_channels == 1:
                ax.imshow(images_to_display[i].reshape(height, width), cmap='gray')
            else:
                ax.imshow(np.transpose(images_to_display[i], (1, 2, 0)))
            ax.axis('off')  # Disable axis
            plt.savefig(self.savepath + f'images_{i}.png')

    
class SSIM():
    def __init__() :
        pass