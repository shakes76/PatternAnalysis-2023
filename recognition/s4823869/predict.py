import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ToPILImage, ToTensor
from torchvision import models
import numpy as np
import random
from matplotlib.pyplot import suptitle, imshow, subplot, axis, show, cm, title
from modules import VectorQuantizer, PixelConvolution, ResidualBlock
from train import PCNN_PATH, LATENT_DIMENSION_SIZE, NUM_EMBEDDINGS, BETA, MODEL_PATH
from dataset import get_ttv, normalise
from skimage.metrics import structural_similarity as ssim
import os

VISUALISATIONS = 5
COLS = 2
MAX_VAL = 1.0
CENTRE = 0.5
GREY = cm.gray
BATCH = 10

def compare(originals, recons):
    ssims = 0
    for i in range(VISUALISATIONS):
        o, r = originals[i], recons[i]
        orig = ToTensor()(ToPILImage()(o)).unsqueeze(0)
        recon = ToTensor()(ToPILImage()(r)).unsqueeze(0)
        sim = ssim(orig, recon, data_range=MAX_VAL)
        ssims += sim
        subplot(VISUALISATIONS, COLS, COLS * i + 1)
        imshow(o + CENTRE, cmap=GREY)
        title("Test Input")
        axis("off")
        subplot(VISUALISATIONS, COLS, COLS * (i + 1))
        imshow(r + CENTRE, cmap=GREY)
        title("Test Reconstruction")
        axis("off")
    suptitle(f"SSIM: {ssims / VISUALISATIONS:.2f}")
    show()

def validate_vqvae(vqvae, test):
    image_inds = random.sample(range(len(test)), VISUALISATIONS)
    images = [test[i] for i in image_inds]
    images = torch.cat(images, dim=0)
    images = images.unsqueeze(1)  # Add channel dimension
    with torch.no_grad():
        recons, _ = vqvae(images)
    recons = recons.squeeze(1)  # Remove channel dimension
    ssim = compare(images, recons)
    avg_ssim = ssim / VISUALISATIONS
    print(f"Average SSIM: {avg_ssim}")

def show_new_brains(priors, samples):
    for i in range(VISUALISATIONS):
        subplot(VISUALISATIONS, COLS, COLS * i + 1)
        imshow(priors[i], cmap=GREY)
        title("PCNN Prior")
        axis("off")
        subplot(VISUALISATIONS, COLS, COLS * (i + 1))
        imshow(samples[i] + CENTRE, cmap=GREY)
        title("Decoded Prior")
        axis("off")
    show()

def show_quantisations(test, encodings, quantiser):
    encodings = encodings[:len(encodings) // 2]  # Throw out half the encodings because of memory constraints
    codebooks = quantiser(torch.from_numpy(encodings))
    codebooks = codebooks.numpy()
    codebooks = codebooks.reshape(encodings.shape[0], encodings.shape[2], encodings.shape[3])

    for i in range(VISUALISATIONS):
        subplot(VISUALISATIONS, COLS, COLS * i + 1)
        imshow(test[i] + CENTRE, cmap=GREY)
        title("Test Image")
        axis("off")
        subplot(VISUALISATIONS, COLS, COLS * (i + 1))
        imshow(codebooks[i], cmap=GREY)
        title("VQ Encoding")
        axis("off")
    show()

def validate_pcnn(vqvae, train_vnce, test):
    pcnn = torch.load(PCNN_PATH)
    priors = torch.zeros((BATCH, *pcnn.input_shape[1:]))
    batch, rows, columns = priors.shape

    for r in range(rows):
        for c in range(columns):
            logs = pcnn(priors)
            m = torch.distributions.Categorical(logits=logs)
            prob = m.sample()
            priors[:, r, c] = prob[:, r, c]

    encoder = vqvae.encoder
    quantiser = vqvae.vector_quantizer
    encoded_out, _ = encoder(test)
    show_quantisations(test, encoded_out, quantiser)
    old_embeds = quantiser.embeddings.clone().detach().cpu()
    priors_onehots = torch.zeros(BATCH, pcnn.input_shape[1], pcnn.input_shape[2], NUM_EMBEDDINGS)
    priors = priors.long()
    priors_onehots.scatter_(3, priors.unsqueeze(3), 1)
    priors_onehots = priors_onehots.float()
    qtised = torch.matmul(priors_onehots, old_embeds.t())
    qtised = qtised.view(BATCH, *encoded_out.shape[1:])
    decoder = vqvae.decoder
    samples, _ = decoder(qtised)
    show_new_brains(priors, samples)

def main():
    tr, te, val = get_ttv()
    test = normalise(te)
    train = normalise(tr)
    vnce = train.var()
    vnce = torch.tensor(vnce)  # Convert to a PyTorch tensor
    vqvae = torch.load(MODEL_PATH)
    validate_vqvae(vqvae, test)
    validate_pcnn(vqvae, vnce, test)

if __name__ == "__main__":
    main()