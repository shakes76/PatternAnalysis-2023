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
from utils import NOISE

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
    
    """
    Use the provided GAN to generate an image.
    """
    def generate_gan(self) :
        self.gan.generator.eval()
        noise = torch.randn(1, NOISE, 1, 1).to(self.device)
        with torch.no_grad():
            self.generated_images = self.gan.generator(noise)
        img = self.generated_images[0][0]
        img = img.cpu()
        img = img.numpy()
        plt.clf()
        plt.tight_layout()
        plt.figure(figsize=(10, 5))
        img = np.clip(img, 0, 1)
        plt.title('GAN Output')
        plt.imshow(img, cmap='gray')
        plt.savefig("./models/predictions/generate_gan.png")

    def quantise(self, input) :
        indices = input.unsqueeze(1)
        encoded = torch.zeros(indices.shape[0], self.vqvae.quantizer.n_embeddings, device = self.device)
        encoded.scatter_(1, indices, 1)
        x_hat = torch.matmul(encoded, self.vqvae.quantizer.embedding.weight).view(1,64,64,64)
        x_hat = x_hat.permute(0,3,1,2).contiguous()
        return x_hat

    def generate_vqvae(self):
        self.vqvae.eval()
        self.generate_gan()
        indice = torch.flatten(self.generated_images[0][0])
        indice = (indice - indice.min()) / (indice.max() - indice.min())
        indice = indice.long()
        quantized = self.quantise(indice)
        out = self.vqvae.decoder(quantized)
        self.generated_images = out
        out = out[0][0].to('cpu').detach().numpy()
        
        plt.clf()
        plt.tight_layout()
        out = np.clip(out, 0, 1)
        plt.figure(figsize=(10, 5))
        plt.title("Generated Image")
        plt.imshow(out, cmap='gray')
        plt.savefig("./models/predicted/generated_vqvae.png")

    
    def ssim(self):
        
        if not self.dataset:
            print('Dataset has not been set')
            return
        
        self.generate_vqvae()
        fake = self.generated_images[0][0].detach().cpu().numpy()  
        
        ssim_scores = []
        for i, (data, _) in enumerate(self.dataset.get_test()):
            real = data[0].cpu().detach().numpy()
            ssim = structural_similarity(
                fake,
                real,
                data_range = fake.max() - fake.min()
            )
            ssim_scores.append(ssim)
        
        avg_ssim = sum(ssim_scores) / len(ssim_scores)
        max_ssim = max(ssim_scores)
        print(f"Average SSIM: {avg_ssim:.4f}\nMax SSIM: {max_ssim:.4f}")
        self.ssim_score = max_ssim
