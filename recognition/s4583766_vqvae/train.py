'''
Training script for the model, including validation, testing, and saving of the model.
Imports the model from modules.py, and the data loader from dataset.py. 
Plots losses and metrics observed during training. 

Author: Sophie Bates, s4583766.
'''

import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import get_dataloaders
from modules import VQVAE, Discriminator, Generator
from predict import validate_batch
from skimage.metrics import structural_similarity as ssim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU...")

# Setup base file paths
PATH = os.getcwd() + '/'
FILE_SAVE_PATH = PATH + 'recognition/s4583766_vqvae/gen_img/'
BASE_DATA_PATH = '/home/groups/comp3710/OASIS/'

# File paths for training, testing, and validation data
TRAIN_DATA_PATH = BASE_DATA_PATH + 'keras_png_slices_train/'
TEST_DATA_PATH = BASE_DATA_PATH + 'keras_png_slices_test/'
VAL_DATA_PATH = BASE_DATA_PATH + 'keras_png_slices_validate/'

# Create new unique directory for this run to store image and model outputs
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_IMG_OUTPUT = FILE_SAVE_PATH + time + "/"
os.makedirs(RUN_IMG_OUTPUT, exist_ok=True)

# Define constant variables/model hyper-parameters
# Many of these are taken from the paper [3], the video
#  [2], the DCGAN tutorial [4] or DCGAN paper [6].
BATCH_SIZE = 32
BATCH_SIZE_GAN = 32
EPOCHS = 1
EPOCHS_GAN = 1
LATENT_SIZE_GAN = 128

N_HIDDEN_LAYERS = 64
N_RESIDUAL_HIDDENS = 32
N_RESIDUAL_LAYERS = 2

EMBEDDINGS_DIM = 64 # Dimension of each embedding in the codebook (embeddings vector)
N_EMBEDDINGS = 512 # Size of the codebook (number of embeddings)

BETA = 0.25
LEARNING_RATE = 1e-3
LR_GAN = 2e-4


def gen_imgs(images, model, device):
    """Generate the reconstructed images for a given batch of images.
    
    Parameters
    ----------
    images : Tensor
        The batch of images to reconstruct, dimensions (batch_size, 1, 256, 256).
    model : VQVAE
        The trained VQ-VAE model.
    device : str
        The device to run the model on.

    Returns
    -------
    Tensor
        The reconstructed images, dimensions (batch_size, 1, 256, 256).
    """
    with torch.no_grad():
        images = images.to(device)
        _, reconstructed_x, _, _ = model(images)

        return reconstructed_x
    
def gen_encodings(model: VQVAE, image):
    """Generate the embeddings for a given image.
    
    Parameters
    ----------
    model : VQVAE
        The trained VQ-VAE model.
    image : Tensor
        The image to encode, dimensions (1, 1, 256, 256).
    """
    with torch.no_grad():
        z_e = model.encoder(image)
        z_e = model.conv1(z_e)
        _, z_q, train_indices_return = model.vector_quantizer(z_e)
        return z_q, train_indices_return

def visualise_embeddings(model: VQVAE, test_dl):
    """Utility function to visualize different snapshots of the model during training.

    Saves the original image, the codebook representation of the image, and the decoded image.
    
    Parameters
    ----------
    model : VQVAE
        The trained VQ-VAE model.
    test_dl : DataLoader
        The test data loader.

    Output: stores each image in the 'gen_img' directory.
    """
    # Take a test image from the dataset and correct the dimensions.
    test_img = next(iter(test_dl))
    test_img = test_img.to(device)
    test_img = test_img[0]
    test_img = test_img.unsqueeze(0)
    test_img.to(device)

    z_q, embeddings = gen_encodings(model, test_img)
    out = embeddings.view(64, 64)

    # Plot the original image before encoding
    plt.imshow(test_img[0][0].detach().cpu().numpy())
    plt.axis("off")
    plt.savefig(
        RUN_IMG_OUTPUT + "codebook_comparison.png", bbox_inches="tight", pad_inches=0
    )
    
    # Plot the codebook representation of the image
    plt.imshow(out.detach().cpu().numpy())
    plt.axis("off")
    plt.savefig(RUN_IMG_OUTPUT + "codebook.png", bbox_inches="tight", pad_inches=0)
    plt.clf()

    # Plot the decoded image
    decoded = model.decoder(z_q)
    print(decoded.shape)
    plt.imshow(decoded[0][0].detach().cpu().numpy())
    plt.axis("off")
    plt.savefig(RUN_IMG_OUTPUT + "decoded.png", bbox_inches="tight", pad_inches=0)

def train_vqvae(vqvae, train_dl, val_dl):
    """
    Train the VQ-VAE model over the training dataset using the loss function 
    described in the paper.

    At each epoch, the model is evaluated on the validation dataset, and the 
    model with the best loss is saved.
    
    The training loop implementation adapted from [2], 
    https://www.youtube.com/watch?v=VZFVUrYcig0.
    
    Parameters
    ----------
    vqvae : VQVAE
        The VQ-VAE model to train.
    train_dl : DataLoader
        The training DataLoader.
    val_dl : DataLoader
        The validation DataLoader.

    Returns
    -------
    VQVAE
        The trained VQ-VAE model.
    
    """
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=LEARNING_RATE)
    
    # Store the original images for comparison across epochs.
    og_imgs = next(iter(train_dl))
    grid = make_grid(og_imgs, nrow=8)
    img_name = f"epoch_0.png"
    save_image(grid, RUN_IMG_OUTPUT + img_name)
    print("Saving", img_name)

    # Initialise the best loss to be updated in loop
    best_loss = float("-inf")

    # flush the standard out print
    sys.stdout.flush()

    avg_train_losses = []
    avg_valid_ssim_losses = []
    # Run training
    for epoch in range(EPOCHS):
        vqvae.train()
        print("Epoch: ", epoch + 1, "\n")
        train_loss = []
        train_steps = 0

        for data in train_dl:
            data = data.to(device)
            optimizer.zero_grad()

            # Get embedding loss, reconstructed image, and the quantized latent vector
            # from the VQVAE model
            embedding_loss, reconstructed_x, z_q, encodings = vqvae(data)

            # Calculate the MSE (reconstruction) loss between the reconstructed image 
            # and the original image. 
            recons_error = F.mse_loss(reconstructed_x, data)

            # Loss is the sum of the embedding loss and the reconstruction error.
            # The embedding loss (v_q) is the MSE between the latent vector and the
            # closest embedding vector in the codebook. It is used to move the 
            # embedding vectors closer to the encoder output.
            loss = embedding_loss + recons_error

            # Backpropagate loss, and update weights.
            loss.backward()
            optimizer.step()

            # Save errors
            train_loss.append(recons_error.item())
            avg_train_losses.append(recons_error.item())

            if train_steps % 100 == 0:
                print("Epoch: {}, Step: {}, Loss: {}".format(epoch, train_steps, np.mean(train_loss[-100:])))
            train_steps += 1
        
        # Change to eval mode for validation
        vqvae.eval()
        
        # Save the model if the average loss is the best we've seen so far.
        avg_train_loss = np.mean(train_loss)

        # Store the model with the best loss
        if avg_train_loss > best_loss:
            best_loss = avg_train_loss
            torch.save(vqvae.state_dict(), RUN_IMG_OUTPUT + "vqvae.pth")

        print("Training loss: {}".format(avg_train_loss))

        # Perform validation on the current model using the 
        # validation dataset
        ssim_losses_val = validate_batch(vqvae, val_dl)
        avg_ssim_loss = np.mean(ssim_losses_val)
        max_ssim_loss = max(ssim_losses_val)
        min_ssim_loss = min(ssim_losses_val)

        print("Validation avg. SSIM loss: {}".format(avg_ssim_loss))
        print("Max SSIM loss: {}".format(max_ssim_loss))
        print("Min SSIM loss: {}".format(min_ssim_loss))

        avg_valid_ssim_losses.append(avg_ssim_loss)
        # avg_train_losses.append(avg_train_loss)

        # Save generated image from the fixed latent vector ('og_imgs')
        gen_img = gen_imgs(og_imgs, vqvae, device)
        grid = make_grid(gen_img.cpu(), nrow=8)
        img_name = f"epoch_{epoch+1}.png"
        save_image(grid, RUN_IMG_OUTPUT + img_name)
        print("Saving", img_name)
        sys.stdout.flush()

    # Plot the average SSIM losses per epoch
    plt.plot(avg_valid_ssim_losses)
    plt.title("SSIM Losses for validation data")
    plt.xlabel("Epoch no.")
    plt.ylabel("SSIM score")
    plt.savefig(os.path.join(RUN_IMG_OUTPUT + "train_ssim_scores.png"))
    plt.clf()

    # Plot the training losses for each batch
    plt.plot(avg_train_losses)
    plt.title("Training reconstruction Losses")
    plt.xlabel("Batch no.")
    plt.ylabel("Reconstruction loss")
    plt.savefig(os.path.join(RUN_IMG_OUTPUT + "train_recon_losses.png"))

    return vqvae


def train_gan(vqvae, train_dl):
    """Train the discriminator and generator in the DCGAN model.

    Parameters
    ----------
    vqvae : VQVAE
        The trained VQVAE model.
    train_dl : DataLoader
        The training data loader.
    
    Returns
    -------
    list
        The generator losses.
    list
        The discriminator losses.
    list
        The real scores.
    list
        The fake scores.

    Note
    ----

    This training loop is adapted from:
        1. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        2. https://www.kaggle.com/code/sushant097/gan-beginner-tutorial-on-celeba-dataset/notebook
        3. UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS - paper (https://arxiv.org/pdf/1511.06434.pdf)
    
    """
    # Initialise the generator and discriminator models
    generator = Generator()
    discriminator = Discriminator()
    criterion = nn.BCELoss()

    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # Create optimizers for generator and discriminator using Adam. 
    # Adam analyzes historical gradients, to adjust the learning rate for each parameter in real-time, 
    # resulting in faster convergence and better performance. It is a combination of RMSProp + Momentum, 
    # it uses moving averages of parameters.
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=LR_GAN, betas=(0.5, 0.999))

    # Define the fixed_latent vector that will be used to compare the generated images across epochs.
    fixed_latent = torch.randn(64, 128, 1, 1, device=device)

    # Generate a before image, and save it.
    og_img_latent = generator(fixed_latent)
    og_imgs = vqvae.decoder(og_img_latent)
    save_image(og_imgs, RUN_IMG_OUTPUT + "gan_epoch_0.png")
    
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    for epoch in range(EPOCHS_GAN):
        print(f"Epoch: {epoch+1}\n ************************************************")
        for real_images in train_dl:
            real_images = real_images.to(device)

            #################### Train discriminator ####################
            # Train on both real images and generated (fake) images. 
            # The training loss is the sum of loss of real + (1-fake). 
            
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
            latent = torch.randn(BATCH_SIZE_GAN, LATENT_SIZE_GAN, 1, 1, device=device)
            fake_images = generator(latent)

            # Pass Fake images through discriminator
            fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
            fake_preds = discriminator(fake_images)
            fake_loss = criterion(fake_preds, fake_targets)
            fake_score = torch.mean(fake_preds).item()

            # Calculate the total discriminator loss. 
            # ---------------------------------------
            # D(G(z)) is the prob that the output of G is real image. 
            # D(x) is the prob that the input to D is real image.

            # Therefore, we want to minimise the loss of D(x), and maximize
            # the loss of D(G(z)), hence: 
            #   loss = log(D(x)) - log(1-D(G(z)))
            loss = real_loss + fake_loss

            # Backpropagate loss, and update weights.
            loss.backward()
            opt_d.step()
            loss_d = loss.item()

            # Clear generator gradients
            opt_g.zero_grad()

            #################### Train Generator ####################
            # Generate fake images that attempt to fool the discriminator. 
            
            # Take the random latent vector that is random Gauss noise, and 
            # decode into an image. 
            latent = torch.randn(BATCH_SIZE_GAN, LATENT_SIZE_GAN, 1, 1, device=device)
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
        vqvae.eval()
        checkpoint_img_dec = vqvae.decoder(checkpoint_img)
        save_image(checkpoint_img_dec, RUN_IMG_OUTPUT + f"gan_epoch_{epoch}.png")

    return losses_g, losses_d, real_scores, fake_scores

def calc_ssim(model: VQVAE, test_dl):
    """Calculate the SSIM score for two sample test images.

    Image 1 is the original image, and image 2 is the reconstructed image after
    being passed through the VQ-VAE model. 
    
    Parameters
    ----------
    model : VQVAE
        The trained VQVAE model.

    test_dl : DataLoader
        The test data loader.
    """
    # Select a test image from the dataset and make it the correct dimensions. 
    test_img = next(iter(test_dl))
    test_img = test_img.to(device)
    test_img = test_img[0]
    test_img = test_img.unsqueeze(0)
    test_img.to(device)

    # Pass the test image through the model to get the embeddings and the reconstructed image.
    z_q, embeddings = gen_encodings(model, test_img)
    out = embeddings.view(64, 64)
    decoded = model.decoder(z_q)
    
    # Convert images to correct format for ssim() method
    image_1 = test_img[0][0].detach().cpu().numpy()
    image_2 = decoded[0][0].detach().cpu().numpy()

    # Calculate SSIM score (using skimage structural similarity)
    ssim_score = ssim(
        image_1, image_2, data_range=image_2.max() - image_2.min(), multichannel=True
    )

    print(ssim_score)

def main():
    model = VQVAE(
        n_hidden_layers=N_HIDDEN_LAYERS, 
        n_residual_hidden_layers=N_RESIDUAL_HIDDENS, 
        n_embeddings=N_EMBEDDINGS,
        embeddings_dim=EMBEDDINGS_DIM, 
        beta=BETA
    )
    model = model.to(device)
    train_dl, test_dl, val_dl = get_dataloaders(TRAIN_DATA_PATH, TEST_DATA_PATH, VAL_DATA_PATH, BATCH_SIZE)
    vqvae = train_vqvae(model, train_dl=train_dl, val_dl=val_dl)

    torch.save(vqvae.state_dict(), RUN_IMG_OUTPUT + "vqvae.pth")
    
    # model = VQVAE(
    #     n_hidden_layers=N_HIDDEN_LAYERS,
    #     n_residual_hidden_layers=N_RESIDUAL_HIDDENS,
    #     n_embeddings=N_EMBEDDINGS,
    #     embeddings_dim=EMBEDDINGS_DIM,
    #     beta=BETA,
    # )
    # ## load in the saved gan model stored at path + 'best-gan.pth'
    # model = Generator()
    # model.to(device)
    # model.load_state_dict(
    #     torch.load(
    #         FILE_SAVE_PATH + "2023-10-22_20-46-12/vqvae.pth",
    #         map_location=torch.device("cpu"),
    #     )
    # )
    vqvae.eval()

    # visualise_embeddings(model, test_dl)

    train_gan(vqvae=vqvae, train_dl=train_dl)

    # TODO: fix this method
    # calc_ssim(model, test_dl)


if __name__ == '__main__':
    main()