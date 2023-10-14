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
        
    def generate(self, image_size=(64, 32, 32)):
        self.model.eval()
        with torch.no_grad():
            # Generate random indices to select embeddings.
            random_indices = torch.randint(high=self.model.quantizer.n_embeddings, size=(self.n, 32, 32)).to(self.device)
            # Retrieve the corresponding embeddings.
            random_embeddings = self.model.quantizer.embedding(random_indices).permute(0, 3, 1, 2)
            # Decode the embeddings into images.
            self.generated_images = self.model.decoder(random_embeddings)

    # def visualise(self):
    #     """
    #     Visualize generated images.

    #     Parameters:
    #     - images (torch.Tensor): Tensor containing images to display.
    #     - self.n (int): Number of images to display.
    #     """
    #     print(self.generated_images.shape)
        # rows = 4 
        # cols = self.n // rows
        # fig, axs = plt.subplots(rows, cols, figsize=(8, 4))
        # axs = axs.ravel()
        # # axs = [axs] if self.n == 1 else axs

        # for i, ax in enumerate(self.generated_images):
        #     axs[i].imshow(self.generated_images[i][0].detach().cpu().numpy() , cmap='gray')
        #     axs[i].axis('off')
        
        # plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        # plt.savefig(self.savepath + f'images_new_{i}.png')
    def visualise(self):
        rows = min(4, self.n)
        cols = self.n // rows
        fig, axs = plt.subplots(rows, cols, figsize=(8, 4))
        axs = np.array(axs).ravel()

        for i, image in enumerate(self.generated_images):
            # Permuting the image tensor so channels are last, and converting to numpy array
            axs[i].imshow(np.transpose(image.detach().cpu().numpy(), (1, 2, 0)))
            axs[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.savefig(self.savepath + f'images_new_{i}.png')
        plt.show()
    
class SSIM():
    def __init__() :
        pass