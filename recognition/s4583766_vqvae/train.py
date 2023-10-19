'''
Training script for the model, including validation, testing, and saving of the model.
Imports the model from modules.py, and the data loader from dataset.py. 
Plots losses and metrics observed during training. 

Sophie Bates, s4583766.
'''

import datetime
import os
import sys

import dataset
import modules
import numpy as np
import torch
import torchvision
from dataset import get_dataloaders, show_images
from modules import VQVAE, Decoder, Encoder
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU...")

# Setup file paths
PATH = os.getcwd() + '/'
FILE_SAVE_PATH = PATH + 'recognition/s4583766_vqvae/gen_img/'
BASE_DATA_PATH = '/home/groups/comp3710/OASIS/'
TRAIN_DATA_PATH = BASE_DATA_PATH + 'keras_png_slices_train/'
TEST_DATA_PATH = BASE_DATA_PATH + 'keras_png_slices_test/'

# Create new unique directory for this run
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_IMG_OUTPUT = FILE_SAVE_PATH + time + "/"
os.makedirs(RUN_IMG_OUTPUT, exist_ok=True)

# Hyper-parameters
BATCH_SIZE = 32
EPOCHS = 3

# Taken from paper and YouTube video
N_HIDDEN_LAYERS = 128
N_RESIDUAL_HIDDENS = 32
N_RESIDUAL_LAYERS = 2

EMBEDDINGS_DIM = 64 # Dimension of each embedding in the codebook (embeddings vector)
N_EMBEDDINGS = 512 # Size of the codebook (number of embeddings)

BETA = 0.25
LEARNING_RATE = 1e-3

def gen_imgs(images, model, device):
    with torch.no_grad():
        images = images.to(device)
        _, x_hat, _ = model(images)
        return x_hat
    
def gen_encodings(model: VQVAE, image):
    with torch.no_grad():
        z_e = model.encoder(image)
        z_e = model.conv1(z_e)
        _, z_q, train_indices_return = model.vector_quantizer(z_e)
        return z_q, train_indices_return

def visualise_embeddings(model: VQVAE, test_dl):
    # load dataset
    test_img = next(iter(test_dl))
    # test_img = test_img
    print(test_img.shape)
    test_img = test_img.to(device)
    test_img = test_img[0]
    test_img = test_img.unsqueeze(0)
    print(test_img.shape)
    test_img.to(device)

    z_q, embeddings = gen_encodings(model, test_img)
    out = embeddings.view(64, 64)
    # out = torch.LongTensor(out)
    # out = out.to)
    # TODO: plot codebook representation - embeddings is a tensor of size (4096)
    ii = out.detach().cpu().numpy()
    plt.imshow(out.detach().cpu().numpy())
    plt.axis("off")
    plt.savefig(RUN_IMG_OUTPUT + "codebook.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

    # plot the test_img image before the encoding
    plt.imshow(test_img[0][0].detach().cpu().numpy())
    plt.axis("off")
    plt.savefig(
        RUN_IMG_OUTPUT + "codebook_comparison.png", bbox_inches="tight", pad_inches=0
    )

    # decoded
    decoded = model.decoder(z_q)
    print(decoded.shape)
    plt.imshow(decoded[0][0].detach().cpu().numpy())
    plt.axis("off")
    plt.savefig(RUN_IMG_OUTPUT + "decoded.png", bbox_inches="tight", pad_inches=0)



def train_vqvae(train_dl):
    vqvae = VQVAE(
        n_hidden_layers=N_HIDDEN_LAYERS, 
        n_residual_hidden_layers=N_RESIDUAL_HIDDENS, 
        n_embeddings=N_EMBEDDINGS,
        embeddings_dim=EMBEDDINGS_DIM, 
        beta=BETA
    )
    vqvae = vqvae.to(device)
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=LEARNING_RATE)
    recon_losses = []
    print(vqvae)    
    
    # Store the original images for comparison
    og_imgs = next(iter(train_dl))
    grid = make_grid(og_imgs, nrow=8)
    img_name = f"epoch_0.png"
    save_image(grid, RUN_IMG_OUTPUT + img_name)
    print("Saving", img_name)

    # Initialise the best loss to be updated in loop
    best_loss = float("-inf")

    vqvae.train()

    # flush the standard out print
    sys.stdout.flush()

    # Run training
    for epoch in range(EPOCHS):
        # print("Epoch: {}".format(epoch+1))
        print("Epoch: ", epoch + 1, "\n")
        train_loss = []
        avg_train_loss = 0
        train_steps = 0
        for data in train_dl:
            data = data.to(device)
            optimizer.zero_grad()

            vq_loss, x_hat, z_q = vqvae(data)

            recons_error = F.mse_loss(x_hat, data)

            loss = vq_loss + recons_error
            loss.backward()
            optimizer.step()

            train_loss.append(recons_error.item())

            if train_steps % 100 == 0:
                print(
                    "Epoch: {}, Step: {}, Loss: {}".format(
                        epoch, train_steps, np.mean(train_loss[-100:])
                    )
                )
            train_steps += 1

        gen_img = gen_imgs(og_imgs, vqvae, device)
        grid = make_grid(gen_img.cpu(), nrow=8)
        # grid = make_grid(og_imgs, nrow=8)
        # show_images(grid, "before")
        img_name = f"epoch_{epoch+1}.png"
        save_image(grid, RUN_IMG_OUTPUT + img_name)
        print("Saving", img_name)
        sys.stdout.flush()

        # Save the model if the average loss is the best we've seen so far.
        avg_train_loss = sum(train_loss) / len(train_loss)

        if avg_train_loss > best_loss:
            best_loss = avg_train_loss
            torch.save(vqvae.state_dict(), RUN_IMG_OUTPUT + "vqvae.pth")

        recon_losses.append(avg_train_loss)
        print("Recon loss: {}".format(avg_train_loss))

def main():
    train_dl, test_dl = get_dataloaders(TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE)
    train_vqvae(train_dl=train_dl)

    model = VQVAE(
        n_hidden_layers=N_HIDDEN_LAYERS,
        n_residual_hidden_layers=N_RESIDUAL_HIDDENS,
        n_embeddings=N_EMBEDDINGS,
        embeddings_dim=EMBEDDINGS_DIM,
        beta=BETA,
    )
    ## load in the saved gan model stored at path + 'best-gan.pth'
    # model = Generator()
    model.to(device)
    model.load_state_dict(
        torch.load(
            FILE_SAVE_PATH + "2023-10-21_13-26-21/vqvae.pth",
            map_location=torch.device("cpu"),
        )
    )
    model.eval()

    visualise_embeddings(model, test_dl)


if __name__ == '__main__':
    main()