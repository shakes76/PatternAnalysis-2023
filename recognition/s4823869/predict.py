import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from torch.distributions import Categorical
from dataset import get_ttv, normalise
from modules import VectorQuantiser, PixelConvolution, ResidualBlock
import numpy as np

VISUALISATIONS = 5
COLS = 2
MAX_VAL = 1.0
CENTRE = 0.5
BATCH = 10
NUM_EMBEDDINGS = 512
PCNN_PATH = "pcnn.pth"
MODEL_PATH = "vqvae.pth"


def compare(originals, recons):
    ssims = 0
    pairs = zip(originals, recons)
    for i, pair in enumerate(pairs):
        o, r = pair
        orig = torch.Tensor(o)
        recon = torch.Tensor(r)
        sim = ssim(orig, recon, data_range=MAX_VAL, multichannel=True)
        ssims += sim
        plt.subplot(VISUALISATIONS, COLS, COLS * i + 1)
        plt.imshow(o + CENTRE, cmap="gray")
        plt.title("Test Input")
        plt.axis("off")
        plt.subplot(VISUALISATIONS, COLS, COLS * (i + 1))
        plt.imshow(r + CENTRE, cmap="gray")
        plt.title("Test Reconstruction")
        plt.axis("off")
    plt.suptitle("SSIM: %.2f" % (ssims / len(originals)))
    plt.show()

def validate_vqvae(vqvae, test):
    image_inds = np.random.choice(len(test), VISUALISATIONS, replace=False)
    images = test[image_inds]
    recons = vqvae(images).cpu().detach().numpy()
    compare(images, recons)

def show_new_brains(priors, samples):
    for i in range(VISUALISATIONS):
        plt.subplot(VISUALISATIONS, COLS, COLS * i + 1)
        plt.imshow(priors[i], cmap="gray")
        plt.title("PCNN Prior")
        plt.axis("off")
        plt.subplot(VISUALISATIONS, COLS, COLS * (i + 1))
        plt.imshow(samples[i] + CENTRE, cmap="gray")
        plt.title("Decoded Prior")
        plt.axis("off")
    plt.show()

def show_quantisations(test, encodings, quantiser):
    encodings = encodings[:len(encodings) // 2]
    flat = encodings.reshape(-1, encodings.shape[-1])
    codebooks = quantiser.code_indices(flat).cpu().numpy().reshape(encodings.shape[:-1])

    for i in range(VISUALISATIONS):
        plt.subplot(VISUALISATIONS, COLS, COLS * i + 1)
        plt.imshow(test[i] + CENTRE, cmap="gray")
        plt.title("Test Image")
        plt.axis("off")
        plt.subplot(VISUALISATIONS, COLS, COLS * (i + 1))
        plt.imshow(codebooks[i], cmap="gray")
        plt.title("VQ Encoding")
        plt.axis("off")
    plt.show()

def validate_pcnn(vqvae, test):
    pcnn = torch.load(PCNN_PATH)  # Load your PCNN model
    priors = torch.zeros(BATCH, *pcnn.input_shape[1:])
    rows, columns = priors.shape

    for r in range(rows):
        for c in range(columns):
            logits = pcnn(priors)
            sampler = Categorical(logits=logits)
            prob = sampler.sample()
            priors[:, r, c] = prob[:, r, c]

    encoder = vqvae.encoder
    quantiser = vqvae.quantiser
    encoded_out = encoder(test)
    show_quantisations(test, encoded_out, quantiser)
    old_embeds = quantiser.embeddings.cpu().detach().numpy()
    pr_onehots = F.one_hot(priors.to(torch.int64), NUM_EMBEDDINGS).cpu().numpy()
    qtised = torch.mm(torch.tensor(pr_onehots, dtype=torch.float32), torch.tensor(old_embeds, dtype=torch.float32).t())
    qtised = qtised.view(-1, *encoded_out.shape[1:])
    decoder = vqvae.decoder
    samples = decoder(qtised)
    show_new_brains(priors.cpu().detach().numpy(), samples.cpu().detach().numpy())

if __name__ == "__main__":
    _, te, _ = get_ttv()
    test = normalise(te)
    vqvae = torch.load(MODEL_PATH)  # Load your VQVAE model
    validate_vqvae(vqvae, test)
    validate_pcnn(vqvae, test)
