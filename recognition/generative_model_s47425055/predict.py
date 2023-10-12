# predict.py

import torch
from modules import VectorQuantizedVAE, compute_ssim
from dataset import test_loader
import matplotlib.pyplot as plt
import numpy as np
from train import MODEL_PATH_TEMPLATE  # Importing the template string directly from train.py

# Load the best epoch value
with open("best_epoch.txt", "r") as file:
    BEST_EPOCH = int(file.read())

DEVICE = torch.device('cuda')
#MODEL_PATH = 'models/checkpoint_epoch{}_vqvae.pt'.format(BEST_EPOCH) # using the BEST_EPOCH to load the best model
MODEL_PATH = MODEL_PATH_TEMPLATE.format(BEST_EPOCH)

# Load the trained model
model = VectorQuantizedVAE(1, 256, 512).to(DEVICE)  # Assuming input_dim=1, dim=256, K=512 from your code
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

def save_image(tensor, filename):
    img = tensor.cpu().clone().detach()
    img = img.numpy().squeeze()
    plt.imshow(img, cmap='gray')
    plt.savefig(filename)


def predict():
    for idx, (x, _) in enumerate(test_loader):
        x = x.to(DEVICE)
        x_tilde, _, _ = model(x)

        # Compute SSIM between original and reconstructed images
        ssim_score = compute_ssim(x, x_tilde)

        print(f"SSIM Score for batch {idx + 1}: {ssim_score:.4f}")
        
        # Save the reconstructed images for visualization
        for j, image in enumerate(x_tilde):
            save_image(image, f'samples3/reconstructed_{idx * len(x) + j + 1}.png')

        # For brevity, let's break after the first batch
        break

if __name__ == '__main__':
    predict()


