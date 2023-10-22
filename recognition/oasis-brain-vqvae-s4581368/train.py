"""
Training module for the VQVAE
Ryan Ward
45813685
"""
import os
import numpy as np
import torch
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from predict import calculate_batch_ssim

def test(data_loader, model, device):
    """
    The test function to validate the model as it is training
    :param Dataloader data_loader: The validation Dataloader
    :param nn.Module model: The VQ-VAE model
    :param str device: The device to run the model on
    :returns: float, float
    """
    with torch.no_grad():
        loss_x, loss_q = 0., 0.
        for images in data_loader:
            images = images.to(device)
            e_loss, x_tilde, _, _= model(images)
            loss_x += F.mse_loss(x_tilde, images)
            loss_q += e_loss
        loss_x /= len(data_loader)
        loss_q /= len(data_loader)

    return loss_x, loss_q

def generate_samples(images, model, device):
    """
    The test function to validate the model as it is training
    :param Tensor images: The images to reconstruct 
    :param nn.Module model: The VQ-VAE model
    :param str device: The device to run the model on
    :returns: Tensor
    """
    with torch.no_grad():
        images = images.to(device)
        _, x_tilde, _, _ = model(images)
    return x_tilde

def train_vqvae(model, save_filename, device, train_loader, val_loader, logger, fixed_images, learning_rate, epochs):
    """"""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = -1
    train_tilde_loss = []
    avg_ssims = []
    avg_ssims.append(0)
    for epoch in range(epochs):
        for images in train_loader:
            images = images.to(device)
            optimizer.zero_grad()

            e_loss, x_tilde, _, _ = model(images)
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

    with open('{0}/model_{1}.pt'.format(save_filename, 'final'), 'wb') as f:
        torch.save(model.state_dict(), f)

    return train_tilde_loss, avg_ssims


def save_samples(index, latent_tensors, sample_dir, net, vqvae):
    """Helper function to save images"""
    fake_images = net.generator(latent_tensors)
    fake_images = vqvae.decoder(fake_images) 
    fake_fname = 'images-{0:0=4d}.png'.format(index)
    save_image(fake_images, os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)

def train_gan(vqvae_model, gan_model, train_loader, device, save_path, epochs, latent_tensors, learning_rate):

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
    gan_model = gan_model.to(device)
    # Adam optimizers for descriminator and generator
    optimizer_descriminator = torch.optim.Adam(gan_model.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(gan_model.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

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
        losses_generator.append(loss_g)
        losses_discriminator.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        # Save generated images
        save_samples(epoch+start_index, latent_tensors, save_path, gan_model, vqvae_model)
    return losses_generator, losses_generator







