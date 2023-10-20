# Training module for the VQVAE
import os
import numpy as np
import torch
from torchvision import transforms, datasets
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from modules import VQVAE, PixelCNN, Decoder, GAN
from dataset import data_loaders, see_data
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

BATCH_SIZE = 32

HIDDEN_LAYERS = 64
RESIDUAL_HIDDEN_LAYERS = 32
RESIDUAL_LAYERS = 2

EMBEDDING_DIMENSION = 64
EMBEDDINGS = 64

BETA = 0.25

LEARNING_RATE = 1e-3
EPOCHS = 15
torch.cuda.empty_cache()

def ssim_batch(x, x_tilde):
    ssims = []
    for i in range(x.shape[0]):
        calculated_ssim = ssim(x[i, 0], x_tilde[i, 0], data_range=(x_tilde[i, 0].max() - x_tilde[i, 0].min()))
        ssims.append(calculated_ssim)
    return ssims


def calculate_batch_ssim(data_loader, model, device):
    with torch.no_grad():
        images = next(iter(data_loader))
        images = images.to(device)
        _, x_tilde, _ = model(images)
        x_tilde = x_tilde.cpu().detach().numpy()
        images = images.cpu().detach().numpy()
        calculated_ssims = ssim_batch(images, x_tilde)
        avg_ssim = sum(calculated_ssims)/len(calculated_ssims)
    return avg_ssim
        

def test(data_loader, model, device):
    with torch.no_grad():
        loss_x, loss_q = 0., 0.
        for images in data_loader:
            images = images.to(device)
            e_loss, x_tilde, x_q = model(images)
            loss_x += F.mse_loss(x_tilde, images)
            loss_q += e_loss
        loss_x /= len(data_loader)
        loss_q /= len(data_loader)

    return loss_x, loss_q

def generate_samples(images, model, device):
    with torch.no_grad():
        images = images.to(device)
        _, x_tilde, _ = model(images)
    return x_tilde

def main():
    now = datetime.now()
    time_rep = now.strftime("%m-%d-%Y-%H_%M_%S")
    logger = SummaryWriter("./logs/{0}".format(time_rep))
    save_filename = "./models/{0}".format(time_rep)
    os.makedirs(save_filename, exist_ok=True)
    train_path = "./keras_png_slices_data/keras_png_slices_train"
    test_path= "./keras_png_slices_data/keras_png_slices_test"
    val_path = "./keras_png_slices_data/keras_png_slices_validate"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_loader, test_loader, val_loader = data_loaders(train_path, test_path, val_path)
    fixed_images = next(iter(train_loader))
    grid = make_grid(fixed_images, nrow=8)
    #plt.imshow(grid.permute(1, 2, 0))
    #plt.show()
    model = VQVAE(HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA)
    if True:

        gan = GAN()
        gan = gan.to(device)
        model.load_state_dict(torch.load("./models/epoch15/model_final.pt"))
        model = model.to(device)
        fixed_images = fixed_images.to(device)
        _, _, quantized, codebook = model(fixed_images)
        q = quantized[0][0].cpu()
        q = q.detach().numpy()
        plt.imshow(q)
        plt.show()
        #train_gan(model, gan, train_loader, device, save_filename, 9)
        return
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    reconstructions = generate_samples(fixed_images, model, device)

    grid = make_grid(reconstructions.cpu(), nrow=8)

#    plt.imshow(grid.permute(1, 2, 0))
#    plt.show()
    logger.add_image('original', grid, 0)
    best_loss = -1
    train_tilde_loss = []
    avg_ssims = []
    avg_ssims.append(0)
    steps = 0
    for epoch in range(EPOCHS):
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            e_loss, x_tilde, x_q = model(images)
            reconstructed_loss = F.mse_loss(x_tilde, images)
            loss = e_loss + reconstructed_loss
            loss.backward()

            optimizer.step()
            train_tilde_loss.append(reconstructed_loss.item()) 

        print("Reconstruction Loss: %.3f" % np.mean(train_tilde_loss[-100:]))

        loss, _ = test(val_loader, model, device)
        batch_avg_ssim = calculate_batch_ssim(val_loader, model, device)
        avg_ssims.append(batch_avg_ssim)
        reconstruction = generate_samples(fixed_images, model, device)
        grid = make_grid(reconstruction.cpu(), nrow=8, range=(-1, 1), normalize=True)
        logger.add_image('reconstruction', grid, epoch + 1)
        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            print("EPOCH{0}\n".format(epoch + 1))

    plt.figure()
    plt.plot(train_tilde_loss)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("./logs/reconstruction_error_training.png")
    plt.figure()
    plt.plot(avg_ssims)
    plt.xlabel("EPOCH")
    plt.ylabel("SSIM")
    plt.savefig("./logs/training_ssims.png")
   
    with open('{0}/model_{1}.pt'.format(save_filename, 'final'), 'wb') as f:
        torch.save(model.state_dict(), f)

    #pixel_cnn = train_pixelcnn(model, train_loader, device, save_filename)
        
    #generate_novel_brains(pixel_cnn, vqvae_decoder, test_loader, device, logger)

    
        #new_model = VQVAE(HIDDEN_LAYERS, RESIDUAL_HIDDEN_LAYERS, EMBEDDINGS, EMBEDDING_DIMENSION, BETA)
    #new_model.load_state_dict(torch.load("./models/10-17-2023-23_34_52/best.pt"))
    #new_model.to(device)
    #new_model.eval()
    #test_trained_model(test_loader, new_model, device)

def train_pixelcnn(vqvae_model: VQVAE, train_loader, device, save_path):
    cnn_model = PixelCNN(1, 64, 1)
    optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
    cnn_model = cnn_model.to(device)
    vqvae_model = vqvae_model.to(device)
    loss = 0
    best_loss = -1
    for epoch in range(EPOCHS):
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            _, _, x_q = vqvae_model(images)

            cnn_output = cnn_model(images)

            loss = F.mse_loss(cnn_output, x_q)

            loss.backward()
            optimizer.step()
            if (epoch == 0) or (loss < best_loss):
                best_loss = loss
            
        print(f"EPOCH: {epoch}, LOSS: {loss}, BEST_LOSS: {best_loss}")

    with open(f"{save_path}/model_cnn_final.pt", 'wb') as f:
        torch.save(cnn_model.state_dict(), f)

    return cnn_model
    
def generate_novel_brains(cnn_model, vqvae_decoder, test_loader, device, logger):
    with torch.no_grad():
        test_images = next(iter(test_loader))
        test_images = test_images.to(device)
        vqvae_decoder = vqvae_decoder.to(device)
        cnn_model.eval()
        cnn_model = cnn_model.to(device)

        x_q = cnn_model(test_images)
        _, _, x_reconstructed = vqvae_decoder(test_images)
        print(x_reconstructed)
        image = x_reconstructed[0].cpu()
        plt.imshow(image)

        plt.show()

def denorm(img_tensors):
    """Function to denormalise images"""
    return img_tensors * 0.5 + 0.5

def save_samples(index, latent_tensors, sample_dir, net, vqvae, show=False):
    """Helper function to save images"""
    fake_images = net.generator(latent_tensors)
    fake_images = vqvae.decoder(fake_images) 
    fake_fname = 'images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)

    if show:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()


def train_gan(vqvae_model, gan_model, train_loader, device, save_path, epochs):

    latent_fixed = torch.rand(64, 128, 1, 1, device=device)
    """Training method for the GAN"""
    torch.cuda.empty_cache()
    # Training data
    start_index = 1
    losses_generator = []
    losses_discriminator = []
    real_scores = []
    fake_scores = []
    loss_d = 0
    loss_g = 0
    real_score = 0
    fake_score = 0

    # Adam optimizers for descriminator and generator
    optimizer_descriminator = torch.optim.Adam(gan_model.discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(gan_model.generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

    # Train
    for epoch in range(epochs):
        for real_images in train_loader:
            # Pass training images to GPU  and train the discriminator and generator
            real_images = real_images.cuda()
            _, _, real_images, _ = vqvae_model(real_images)
            # Train discriminator
            """Train the discriminator"""
            # Level the gradients
            optimizer_descriminator.zero_grad()
            
            # Find the predictions of the  real set of images
            real_preds = gan_model.discriminator(real_images)
            real_targets = torch.ones(real_images.size(0), 1, device=device)
            real_loss = F.binary_cross_entropy(real_preds, real_targets)
            # Should be very close to 1
            real_score = torch.mean(real_preds).item()

            # Create a latent space to generate a fake image
            latent = torch.randn(32, 128, 1, 1, device=device)
            # Create fake images
            fake_images = gan_model.generator(latent)
            
            # Predict to see if the discriminator can determine if image is fake
            fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
            fake_preds = gan_model.discriminator(fake_images)
            fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
            fake_score = torch.mean(fake_preds).item()

            # Update discriminator weights
            loss = real_loss + fake_loss
            loss.backward()
            optimizer_descriminator.step()
            loss_d = loss.item()
            # Train generator

            optimizer_generator.zero_grad()
    
            # Create latent space
            latent = torch.randn(32, 128, 1,1, device=device)

            # generate fake images
            fake_images = gan_model.generator(latent)

            # pass outputs through discriminator
            preds = gan_model.discriminator(fake_images)
            targets = torch.ones(32, 1, device=device)
            loss = F.binary_cross_entropy(preds, targets)

            # Update generator 
            loss.backward()
            optimizer_generator.step()
            loss_g = loss.item()

        # Record losses & scores
        losses_generator.append(losses_generator)
        losses_discriminator.append(losses_discriminator)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        # Save generated images
        save_samples(epoch+start_index, latent_fixed, save_path, gan_model, vqvae_model, show=False)
if __name__ == '__main__':
    main()





