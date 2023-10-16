import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class TestVQVAE() :
    def __init__(self, model: nn.Module, dataset, savepath='./models/vqvae', device='cpu') :
        self.model = model
        self.dataset = dataset
        self.savepath = savepath
        self.device = device


    def reconstruct(self, path: None, show = False) :
        x, label = next(iter(self.dataset.get_test()))
        x = x.to(self.device)
        x = self.model.encoder(x)
        x = self.model.conv(x)
        _, x_hat, _, embeddings, _ = self.model.quantizer(x)
        x_recon = self.model.decoder(x_hat)

        rows = 4 
        batch = x_recon.shape[0]
        cols = batch // rows

        fig, axs = plt.subplots(rows, cols, figsize=(8, 4)) 
        axs = axs.ravel()

        for i in range(x_recon.shape[0]):
            axs[i].imshow(x_recon[i][0].detach().cpu(), cmap='gray')
            axs[i].axis('off')

        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        if path :
            plt.savefig(path)
        else :
            plt.savefig(self.savepath + "/reconstructed.png")
            
        if show:        
            plt.show()

