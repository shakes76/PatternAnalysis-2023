import torch
from torch import optim, nn
from modules import VQVAE, VQVAETrainer, PixelCNN
from dataset import BrainSlicesDataset, get_image_slices
import matplotlib.pyplot as plt

# This function trains the VQ-VAE model.
def train_vqvae(vqvae, train_loader, num_epochs, learning_rate, test_samples, recon_losses, vq_losses, perplexities):
    # Set up the optimizer for training (Adam in this case).
    optimizer = optim.Adam(vqvae.parameters(), lr=learning_rate)

    # Loop through each epoch.
    for epoch in range(num_epochs):
        # Loop through each batch of images from the DataLoader.
        for batch_idx, images in enumerate(train_loader):
            images = images.to(device)  # Transfer images to the GPU if available.

            # Zero the gradients.
            optimizer.zero_grad()

            # Forward pass through the VQ-VAE.
            x_recon, perplexity, loss = vqvae(images)

            # Compute reconstruction and VQ losses.
            recon_loss_value = F.mse_loss(x_recon, images) / vqvae.train_variance
            vq_loss_value = loss - recon_loss_value

            # Record the losses and perplexity for plotting later.
            recon_losses.append(recon_loss_value.item())
            vq_losses.append(vq_loss_value.item())
            perplexities.append(perplexity.item())

            # Backward pass.
            loss.backward()

            # Update the weights.
            optimizer.step()

        # At the end of each epoch, visualize some reconstructed images.
        with torch.no_grad():
            reconstructions, _, _ = vqvae(test_samples)
            visualize_reconstructions(test_samples.cpu(), reconstructions.cpu())

            # Save the generated images
            save_path = os.path.join(OUTPUT_DIR, f"{epoch}.png")
            save_image(reconstructions, save_path)

        # Print the loss for the current epoch.
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # At the end of training, plot the recorded losses and perplexity.
    plt.figure(figsize=(10,5))
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.plot(vq_losses, label='VQ Loss')
    plt.legend()
    plt.title('Losses over Training')
    plt.xlabel('Training Iterations')
    plt.ylabel('Loss Value')
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(perplexities)
    plt.title('Perplexity over Training')
    plt.xlabel('Training Iterations')
    plt.ylabel('Perplexity')
    plt.show()

    # Visualize the histogram of encoding indices.
    with torch.no_grad():
        _, _, _, encoding_indices = vqvae.vqvae.quantize(vqvae.vqvae.encoder(test_samples))
        encoding_indices = encoding_indices.flatten().cpu().numpy()

    plt.figure(figsize=(10,5))
    plt.hist(encoding_indices, bins=np.arange(vqvae.vqvae.quantize.num_embeddings+1)-0.5, rwidth=0.8)
    plt.title('Histogram of Encoding Indices')
    plt.xlabel('Encoding Index')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(vqvae.vqvae.quantize.num_embeddings))
    plt.show()

        # Print the recorded losses and perplexities
    print("Reconstruction Losses:", recon_losses)
    print("VQ Losses:", vq_losses)
    print("Perplexities:", perplexities)

# This function trains the PixelCNN model.
def train_pixelcnn(pixelcnn, train_loader, num_epochs, learning_rate):
    optimizer = optim.Adam(pixelcnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Loop through each epoch.
    for epoch in range(num_epochs):
        # Loop through each batch of images from the DataLoader.
        for images in train_loader:
            images = images.to(device)  # Transfer images to the GPU if available.

            # Zero the gradients.
            optimizer.zero_grad()

            # Forward pass through the PixelCNN.
            logits = pixelcnn(images)

            # Compute the loss.
            loss = criterion(logits, images.squeeze(1).long())

            # Backward pass.
            loss.backward()

            # Update the weights.
            optimizer.step()

        # Print the loss for the current epoch.
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
