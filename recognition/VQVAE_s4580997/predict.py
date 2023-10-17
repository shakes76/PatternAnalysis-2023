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
    
    def generate_gan(self, num_images = 1) :
        self.gan.generator.eval()
        noise = torch.randn(1, 100, 1, 1).to(self.device)
        with torch.no_grad():
            self.generated_images = self.gan.generator(noise)
        self.visualise(num_images=1, show = False, save = True, savename="gan_generated")

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
    
    def generate_pixelcnn_vqvae(self, pixelcnn, num_images=1):
        self.vqvae.eval()
        pixelcnn.eval()

        latent_map_shape = (num_images, 128, *self.img_size)
        latent_map = torch.zeros(latent_map_shape, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i in range(self.img_size[0]):
                for j in range(self.img_size[1]):
                    pixel_probs = pixelcnn(latent_map[:, :, :i+1, :j+1].float())
                    sampled_pixel = torch.multinomial(F.softmax(pixel_probs[:, :, i, j], dim=1), 1).squeeze(1)
                    latent_map[:, :, i, j] = sampled_pixel
            
            embedding_result = self.vqvae.quantizer.embedding(latent_map)
            print(f"Latent map shape: {latent_map.shape}")
            print(f"Embedding result shape: {embedding_result.shape}")

            # Taking the mean along the channel dimension to get a [batch_size, 32, 32, 64] tensor
            random_embeddings = embedding_result.mean(dim=1)
            print(f"Random embeddings shape: {random_embeddings.shape}")

            self.generated_images = self.vqvae.decoder(random_embeddings.permute(0, 3, 1, 2))

        self.visualise(num_images=num_images, show=False, save=True, savename="pixelcnn_vqvae_generated")







    def generate_image(self, pixelcnn, num_images=1):
        self.vqvae.eval()
        pixelcnn.eval()

        # Assuming your embeddings are of shape: [num_embeddings, embedding_dim]
        embedding_dim = self.vqvae.quantizer.embedding.weight.shape[1]

        # Initialize latent map
        latent_map_shape = (num_images, *self.img_size)
        latent_map = torch.zeros(latent_map_shape, dtype=torch.long, device=self.device)

        with torch.no_grad():
            for i in range(self.img_size[0]):
                for j in range(self.img_size[1]):
                    # Get the probabilities for the next pixel from PixelCNN
                    pixel_probs = pixelcnn(latent_map)
                    
                    # Sample the next pixel value
                    sampled_pixel = torch.multinomial(F.softmax(pixel_probs[:, :, i, j], dim=1), 1).squeeze(1)
                    
                    # Update the latent map
                    latent_map[:, i, j] = sampled_pixel
            
            # Retrieve the corresponding embeddings
            random_embeddings = self.vqvae.quantizer.embedding(latent_map).permute(0, 3, 1, 2)
            self.generated_images = self.vqvae.decoder(random_embeddings)

        self.visualise(num_images=num_images, show = False, save = True, savename="pixelcnn_vqvae_generated")

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
        print(model)
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
