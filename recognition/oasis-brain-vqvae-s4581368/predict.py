# Prediction framework for the VQVAE
import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from modules import VQVAE, PixelCNN, Decoder, GAN
from train import HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA
from dataset import data_loaders, see_data
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import argparse

def generate_vqvae_samples(model, test_loader, device):
    with torch.no_grad():
        images = next(iter(test_loader))
        images = images.to(device)
        _, x_tilde, _, _ = model(images)

    return x_tilde

    

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_path= "./keras_png_slices_data/keras_png_slices_test"
    vq_path = args.vq_path
    vq_vae = VQVAE(HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA)
    vq_vae.load_state_dict(torch.load(vq_path))
    vq_vae.to(device)
    vq_vae.eval()
    _, test_loader, _ = data_loaders(test_path, test_path, test_path)
    generate_vqvae_samples(vq_vae, test_loader, device)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="predict",
        description="Specify the prediction parameters for Task 8"
    )
    parser.add_argument('vq_path', type=str, help="The VQ-VAE pretrained model to generated predictions")
    parser.add_argument('--gan_path', type=str, help="The GAN pretrained model to generated predictions")
    args = parser.parse_args()
    main(args)

