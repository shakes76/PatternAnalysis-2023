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
    def __init__(self, vqvae, gan, dataset, device='cpu', savepath = './models/predictions/', img_size=(32, 32)) -> None:
        self.vqvae = vqvae
        self.gan = gan
        self.dataset = dataset

        self.device = device
        self.savepath = savepath

        self.generated_images = None
        self.ssim_score = None

        self.img_size=img_size
    
    def generate_gan(self, num_images = 1) :
        self.gan.generator.eval()
        noise = torch.randn(num_images, self.gan.latent_dim, 1, 1).to(self.device)
        with torch.no_grad():
            self.generated_images = self.gan.generator(noise)
                    
        self.visualise(num_images=num_images, show = False, save = True, savename="gan_generated")

    def generate_vqvae(self, num_images=1):
        self.vqvae.eval()
        with torch.no_grad():
            random_indices = torch.randint(
                high=self.vqvae.quantizer.n_embeddings, 
                size=(num_images, *self.img_size)
            ).to(self.device)
            
            random_embeddings = self.vqvae.quantizer.embedding(random_indices).permute(0, 3, 1, 2)
            self.generated_images = self.vqvae.decoder(random_embeddings)
        
        self.visualise(num_images=num_images, show = False, save = True, savename="vqvae_generated")

    def visualise(self, num_images = 4, show = True, save = True, savename="out"):
        rows = min(4, num_images)
        cols = num_images // rows
        fig, axs = plt.subplots(rows, cols, figsize=(8, 4))
        axs = np.array(axs).ravel()

        for i, image in enumerate(self.generated_images):
            axs[i].imshow(np.transpose(image.detach().cpu().numpy(), (1, 2, 0)))
            axs[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        if save:
            plt.savefig(self.savepath + f'{savename}_{i}.png')
        if show:
            plt.show()
    
    def ssim(self, model='vqvae', num_images=32):
        
        if not self.dataset:
            print('Dataset has not been set')
            return
        
        if model == 'gan' :
            self.generate_gan()
        else :
            self.generate_vqvae()

        gen_img = self.generated_images[0].detach().cpu().numpy()  
        gen_img = np.transpose(gen_img, (1, 2, 0)) 
        
        ssim_scores = []

        for i, (data, _) in enumerate(self.dataset.get_test()):
            
            if i >= num_images:
                break
            real_img = data[0].numpy() 
            real_img = np.transpose(real_img, (1, 2, 0))

            ssim = structural_similarity(
                gen_img,
                real_img,
                multichannel=True,
                win_size=3,
                data_range=1.0,
            )
            ssim_scores.append(ssim)
        
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        max_ssim = max(ssim_scores)
        print(f"Average SSIM: {avg_ssim:.4f}\nMax SSIM: {max_ssim:.4f}")
        self.ssim_score = max_ssim
