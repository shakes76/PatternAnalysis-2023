##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

class Predict() :
    def __init__(self, model, num_images = 1, device='cpu', savepath = '', dataset = None) -> None:
        self.model = model
        self.n = num_images
        self.device = device
        self.savepath = savepath
        self.dataset = dataset

        self.generated_images = None
        self.ssim_score = None
        
    def generate(self, num = None):
        self.model.eval()
        with torch.no_grad():
            # Generate random indices to select embeddings.
            random_indices = torch.randint(high=self.model.quantizer.n_embeddings, size=(self.n, 32, 32)).to(self.device)
            # Retrieve the corresponding embeddings.
            random_embeddings = self.model.quantizer.embedding(random_indices).permute(0, 3, 1, 2)
            # Decode the embeddings into images.
            self.generated_images = self.model.decoder(random_embeddings)

    def visualise(self, show = True, save = True):
        rows = min(4, self.n)
        cols = self.n // rows
        fig, axs = plt.subplots(rows, cols, figsize=(8, 4))
        axs = np.array(axs).ravel()

        for i, image in enumerate(self.generated_images):
            # Permuting the image tensor so channels are last, and converting to numpy array
            axs[i].imshow(np.transpose(image.detach().cpu().numpy(), (1, 2, 0)))
            axs[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        if save:
            plt.savefig(self.savepath + f'images_new_{i}.png')
        if show:
            plt.show()
    
    def ssim(self):
        if not self.dataset:
            print('Dataset has not been set')
            return
        
        self.generate()
        gen_img = self.generated_images[0].detach().cpu().numpy()  
        gen_img = np.transpose(gen_img, (1, 2, 0)) 
        
        ssim_scores = []

        for i, (data, _) in enumerate(self.dataset.get_test()):
            
            if i >= self.n:
                break
            real_img = data[0].numpy()  # Taking the first image in the batch
            real_img = np.transpose(real_img, (1, 2, 0))  # HxWxC

            print(real_img.shape)
            print(gen_img.shape)
            assert real_img.shape == gen_img.shape, "Image shapes do not match!"

            ssim = structural_similarity(
                gen_img,
                real_img,
                multichannel=True,
                win_size=3,
                data_range=1.0,
            )
            ssim_scores.append(ssim)
            print(f"SSIM between generated image and test image {i}: {ssim:.4f}")
        
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        print(f"Average SSIM: {avg_ssim:.4f}")
        self.ssim_score = avg_ssim
        return ssim_scores