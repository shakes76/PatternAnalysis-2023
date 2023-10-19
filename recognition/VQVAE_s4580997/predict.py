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
import torch.nn.functional as F

class Predict() :
    def __init__(self, vqvae, gan, dataset, device='cpu', savepath = './models/predictions/', img_size=64) -> None:
        self.vqvae = vqvae
        self.gan = gan
        self.dataset = dataset

        self.device = device
        self.savepath = savepath

        self.generated_images = None
        self.ssim_score = None

        self.img_size=(img_size, img_size)
    
    def generate_gan(self) :
        self.gan.generator.eval()
        noise = torch.randn(1, 100, 1, 1).to(self.device)
        with torch.no_grad():
            self.generated_images = self.gan.generator(noise)
        img = self.generated_images[0][0] #Get single image for plotting 64 x 64
        img = img.cpu()
        img = img.numpy()
        plt.clf()
        plt.tight_layout()
        plt.figure(figsize=(10, 5))
        img = np.clip(img, 0, 1)
        plt.title('GAN Output')
        plt.imshow(img, cmap='gray')
        plt.savefig("./models/predictions/generate_gan.png")

    def generate_vqvae(self, num_images=1):
        self.vqvae.eval()
        self.generate_gan()
        indice = torch.flatten(self.generated_images[0][0])
        indice = (indice - indice.min()) / (indice.max() - indice.min())
        indice.long()
        indice = indice.unsqueeze(0)
        quantized = self.vqvae.quantizer(indice)
        self.generated_images = self.vqvae.decoder(quantized)
    

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
        
        ssim_scores = []
        for i, (data, _) in enumerate(self.dataset.get_test()):
            
            if i >= num_images:
                break
            real_img = data[0].cpu().detach().numpy() 
            print(gen_img.shape)
            print(real_img.shape)
            ssim = structural_similarity(
                gen_img,
                real_img,
                multichannel=True,
                win_size=3,
                data_range = gen_img.max() - gen_img.min()
            )
            ssim_scores.append(ssim)
        
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        max_ssim = max(ssim_scores)
        print(f"Average SSIM: {avg_ssim:.4f}\nMax SSIM: {max_ssim:.4f}")
        self.ssim_score = max_ssim
