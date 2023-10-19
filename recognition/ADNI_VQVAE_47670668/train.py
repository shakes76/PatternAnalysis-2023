import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim
from sklearn.cluster import KMeans

from dataset import train_dataloader, val_dataloader, test_loader
from modules import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def visualize_reconstructions(original_images, reconstructed_images, num_samples=10):
    # This function assumes the images are tensors with shape [batch_size, channels, height, width]
    
    num_samples = min(num_samples, original_images.size(0))  # Ensure num_samples is within bounds
    
    _, axs = plt.subplots(2, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        axs[0, i].imshow(original_images[i].permute(1, 2, 0).cpu().numpy())
        axs[1, i].imshow(reconstructed_images[i].detach().permute(1, 2, 0).cpu().numpy())
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.show()

# initialize all variables for training
num_training_updates = 10000
num_epochs = 5  

num_hiddens = 256
num_residual_hiddens = 64  
num_residual_layers = 2

embedding_dim = 64

num_embeddings = 128  

commitment_cost = 0.1  

decay = 1e-5 

learning_rate = 1e-4

encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)

vq_vae = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)

pre_vq_conv1 = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)

model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

torch.autograd.set_detect_anomaly(True)

def train_step(image, label): 
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Move data to device
    image = image.to(device)
    label = label.to(device)

    # Forward pass
    model_output = model(image)
    loss = model_output['loss']

    # Calculate SSIM
    ssim_value = ssim(model_output['x_recon'], image, data_range=1.0)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    return model_output, ssim_value.item()

for epoch in range(num_epochs):  # Added epoch loop

    # Reset training metrics at the start of each epoch
    train_recon_errors = []
    train_ssim_values = []
    
    for step_index, (image, label) in enumerate(train_dataloader): # Updated data unpacking

        train_results, ssim_value = train_step(image, label)
        train_ssim_values.append(ssim_value)
        train_recon_errors.append(train_results['recon_error'].item())


        if (step_index + 1) % 100 == 0:  # Adjust frequency as needed
            print('Epoch %d/%d - Step %d ' % (epoch + 1, num_epochs, step_index + 1) +
                  ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                  ('ssim: %.3f' % np.mean(train_ssim_values[-100:]))) 

        if step_index == num_training_updates:
            break

    scheduler.step()

    # Visualization logic
    with torch.no_grad():
        reconstructed_images = train_results['x_recon']
        reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
    visualize_reconstructions(image, reconstructed_images)

    # After training loop, begin validation
    model.eval()  # Switch to evaluation mode

    # Initialize validation metrics
    
    val_losses = []
    val_recon_errors = []
    val_perplexities = []
    val_vqvae_loss = []
    
    with torch.no_grad():  # Disable gradient computation during validation
        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device)
            val_results = model(image)
            val_recon_errors.append(val_results['recon_error'].item())

    # Print validation metrics
    print(f"Epoch {epoch + 1}/{num_epochs} - "
          f"recon_error: {np.mean(val_recon_errors):.3f}," )

    model.train()  # Switch back to training mode


# Evaluate ssim on test set
model.eval()  # Set the model to evaluation mode
average_ssim = 0
with torch.no_grad():
    for inputs, _ in test_loader:  
        inputs = inputs.to(device)  

        # Forward pass
        output_dict = model(inputs)

        # Extract the reconstructed images tensor
        reconstructed_images = output_dict['x_recon']  

        # Calculate SSIM
        current_ssim = ssim(reconstructed_images, inputs, data_range=1.0)
        average_ssim += current_ssim.item()

# Average the SSIM over all batches
average_ssim = average_ssim / len(test_loader)

print(f"Average SSIM on test set: {average_ssim:.4f}")