'''
Training script for the model, including validation, testing, and saving of the model.
Imports the model from modules.py, and the data loader from dataset.py. 
Plots losses and metrics observed during training. 

Sophie Bates, s4583766.
'''

import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_dataloaders
from modules import VQVAE, Discriminator, Generator
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from skimage.metrics import structural_similarity as ssim


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
BATCH_SIZE_GAN = 256
EPOCHS = 3
EPOCHS_GAN = 20

# Taken from paper and YouTube video
N_HIDDEN_LAYERS = 128
N_RESIDUAL_HIDDENS = 32
N_RESIDUAL_LAYERS = 2

EMBEDDINGS_DIM = 64 # Dimension of each embedding in the codebook (embeddings vector)
N_EMBEDDINGS = 512 # Size of the codebook (number of embeddings)

BETA = 0.25
LEARNING_RATE = 1e-3
LR_GAN = 2e-4


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

def train_gan(model, train_dl):
    # define dataset

    # Create ds and dl
    transform = tfs.Compose([tfs.ToTensor()])
    gan_ds = GanData(model, transform)
    gan_dl = DataLoader(gan_ds, batch_size=BATCH_SIZE_GAN)
    generator = Generator()
    discriminator = Discriminator()

    # optimizer_gen = torch.optim.Adam(generator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
    # optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    fixed_latent = torch.randn(BATCH_SIZE_GAN, 100, 1, 1, device=device)
    # save_samples(0, fixed_latent)

    # og_imgs = next(iter(train_dl))
    # og_imgs = og_imgs.to(device)

    # create new subfolder thats title is the current time
    og_imgs = generator(fixed_latent)
    save_image(og_imgs, RUN_IMG_OUTPUT + "gan_epoch_0.png")
    # print("Saving", img_name)
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers for generator and discriminator using Adam. Adam analyzes historical gradients, to adjust the learning rate for each parameter in real-time, resulting in faster convergence and better performance.
    # Adam is a combination of RMSProp + Momentum, it uses moving averages of parameters.
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))

    for epoch in range(EPOCHS_GAN):
        for real_images in train_dl:
            real_images = real_images.to(device)
            # Train discriminator
            # loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            # Clear discriminator gradients
            opt_d.zero_grad()

            # Pass real images through  discriminator, so discriminator can
            # learn what they look like.
            real_preds = discriminator(real_images)
            real_targets = torch.ones(real_images.size(0), 1, device=device)
            real_loss = criterion(real_preds, real_targets)
            real_score = torch.mean(real_preds).item()

            # Generate fake images. Latent space = representation of compressed data.
            # Use the same seed every time, so we can compare the generated images.
            latent = torch.randn(BATCH_SIZE_GAN, 100, 1, 1, device=device)
            fake_images = generator(latent)

            # Pass Fake images through discriminator
            fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
            fake_preds = discriminator(fake_images)
            fake_loss = criterion(fake_preds, fake_targets)
            fake_score = torch.mean(fake_preds).item()

            loss = real_loss + fake_loss
            # Backpropagate loss, and update weights.
            loss.backward()
            opt_d.step()
            loss_d = loss.item()

            opt_g.zero_grad()

            # Generate a batch of fake images.
            latent = torch.randn(BATCH_SIZE_GAN, 100, 1, 1, device=device)
            fake_images = generator(latent)

            # Try to fool the discriminator
            preds = discriminator(fake_images)
            # Set labels to '1' to indicate they're fake.
            targets = torch.ones(BATCH_SIZE_GAN, 1, device=device)
            # Minimize loss of D(G(z)).
            loss_g = criterion(preds, targets)

            # Backpropagate loss, and update weights.
            loss_g.backward()
            loss_g = loss_g.item()
            opt_g.step()

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print(
            "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, EPOCHS, loss_g, loss_d, real_score, fake_score
            )
        )
        # Save generated images
        # save_samples(epoch+start_idx, fixed_latent, show=False)
        # save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
        checkpoint_img = generator(fixed_latent)
        save_image(denorm(checkpoint_img), RUN_IMG_OUTPUT + f"gan_epoch_{epoch}.png")

    return losses_g, losses_d, real_scores, fake_scores


def calc_ssim(model, test_dl):
    # Calculate SSIM for two images
    test_img = next(iter(test_dl))

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

    # decoded
    decoded = model.decoder(z_q)
    # print(decoded.shape)
    # plt.imshow(decoded[0][0].detach().cpu().numpy())
    # plt.axis('off')
    # plt.savefig(RUN_IMG_OUTPUT + 'decoded.png', bbox_inches='tight', pad_inches=0)

    image_1 = test_img[0][0].detach().cpu().numpy()
    image_2 = decoded[0][0].detach().cpu().numpy()

    # calculate ssim
    ssim_score = ssim(
        image_1, image_2, data_range=image_2.max() - image_2.min(), multichannel=True
    )
    print(ssim_score)

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

    train_gan(model, train_dl)

    calc_ssim(model, test_dl)


if __name__ == '__main__':
    main()