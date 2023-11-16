# train.py

import torch
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
from itertools import cycle
import torch.nn.functional as F
from torch.distributions.normal import Normal
from modules import VectorQuantizedVAE, compute_ssim, weights_init, to_scalar, plot_losses_and_scores, generate_samples
from dataset import train_loader, val_loader, test_loader

# Constants
BATCH_SIZE = 32
#N_EPOCHS = 401
N_EPOCHS = 346
PRINT_INTERVAL = 100
DATASET_PATH = './OASIS'
NUM_WORKERS = 1
INPUT_DIM = 1
DIM = 256
K = 512
LAMDA = 1
LR = 1e-3
DEVICE = torch.device('cuda')

# Global best epoch and model path
BEST_EPOCH = 0
MODEL_PATH_TEMPLATE = 'samples6/checkpoint_epoch{}_vqvae.pt'

# Constants for determining the importance of SSIM and reconstruction loss
ALPHA = 0.5  # weight for SSIM, range [0, 1]
BETA = 1 - ALPHA  # weight for reconstruction loss
BEST_METRIC = -999  # initial value for the combination metric
BEST_SSIM = 0  # just for logging purposes
BEST_RECONS_LOSS = 999  # just for logging purposes
save_interval = 15

#directory creation
train_losses_epoch = []
val_losses = []
ssim_scores = []

# Directories
#Path('models2').mkdir(exist_ok=True)
Path('samples6').mkdir(exist_ok=True)
Path('models6').mkdir(exist_ok=True)

# Model setup
model = VectorQuantizedVAE(INPUT_DIM, DIM, K).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR, amsgrad=True)
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)  # for example, reduce LR every 10 epochs by 10%

"""
train(void)
-----------
This function trains the VQ-VAE model using a given DataLoader for the training dataset.
It computes the reconstruction loss, vector quantization (VQ) loss, and commitment loss
during each training iteration. The function also calculates the negative log-likelihood (NLL)
of the model's output, which represents the log-likelihood of the input data under the model's
Gaussian assumption.

Inputs: None
Outputs: None
"""
def train():
    model.train() # Set the model to training mode
    # initialise losses and count of batches processed in this epoch
    total_loss_recons = 0.0
    total_loss_vq = 0.0
    num_batches = 0

    for batch_idx, (x, _) in enumerate(train_loader):
        start_time = time.time() # Record the start time for batch processing
        x = x.to(DEVICE) # Move the input data to the specified device (e.g., GPU)

        opt.zero_grad() #  Zero out the gradients in the optimizer
        # Forward pass through the model to obtain reconstructed data (x_tilde) and latent codes (z_e_x, z_q_x)
        x_tilde, z_e_x, z_q_x = model(x)
        z_q_x.retain_grad() # Retain gradients of z_q_x for backpropagation

        # Compute the mean squared error (MSE) loss between x_tilde and the original input x
        loss_recons = F.mse_loss(x_tilde, x)
        loss_recons.backward(retain_graph=True) # Backpropagate the reconstruction loss
        total_loss_recons += loss_recons.item() # Accumulate the reconstruction loss

        # Straight-through estimator: Backpropagate gradients from z_q_x to z_e_x
        z_e_x.backward(z_q_x.grad, retain_graph=True)

        # Vector quantization (VQ) objective: Compute MSE loss between z_q_x and z_e_x (detached)
        model.codebook.zero_grad() # Zero out gradients in the codebook
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        loss_vq.backward(retain_graph=True) # Backpropagate the VQ loss

        # Commitment objective: Compute MSE loss between z_e_x and z_q_x (detached) with a weighting factor LAMDA
        loss_commit = LAMDA * F.mse_loss(z_e_x, z_q_x.detach())
        loss_commit.backward() # Backpropagate the commitment loss
        opt.step() # Update model parameters using the optimizer

        N = x.numel() # Calculate the total number of elements in the input
        nll = Normal(x_tilde, torch.ones_like(x_tilde)).log_prob(x) # Compute negative log-likelihood
        log_px = nll.sum() / N + np.log(128) - np.log(K * 2) #  Calculate log likelihood per dimension
        log_px /= np.log(2)

        if (batch_idx + 1) % PRINT_INTERVAL == 0:
            print('\tIter [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                batch_idx * len(x), len(train_loader.dataset),
                PRINT_INTERVAL * batch_idx / len(train_loader),
                np.mean(train_losses_epoch[-PRINT_INTERVAL:], axis=0),
                time.time() - start_time
            ))
        total_loss_vq += loss_vq.item() # Accumulate the VQ loss
        num_batches += 1

    if num_batches > 0:
        avg_loss_recons = total_loss_recons / num_batches # Calculate the average reconstruction loss
        avg_loss_vq = total_loss_vq / num_batches # Calculate the average VQ loss
    else:
        avg_loss_recons = 0
        avg_loss_vq = 0
    
    train_losses_epoch.append((avg_loss_recons, avg_loss_vq)) # Append the average losses for this epoch
    print('Epoch Loss: Recons {:.4f}, VQ {:.4f}'.format(avg_loss_recons, avg_loss_vq))

"""
validate(void)
-------------
This function performs model validation on a validation dataset using the VQ-VAE model.
It calculates validation losses, including the reconstruction loss (mean squared error) and
the vector quantization (VQ) loss. Additionally, it computes the Structural Similarity Index (SSIM)
for each batch of validation data and accumulates these scores.

Input: None
Output: combined_metric (float): Combined evaluation metric, which is a weighted sum of 
    SSIM and reconstruction loss.

"""
def validate():
    model.eval()  # Switch to evaluation mode
    # Initialize variables for validation loss, SSIM scores, and batch counts
    val_loss = []
    ssim_accum = 0.0  # Accumulator for SSIM scores
    batch_count = 0   # Counter for batches
    total_loss_recons = 0.0
    total_loss_vq = 0.0
    num_batches = 0
    # Disable gradient computation during validation
    with torch.no_grad():  # No gradient required for validation
        # Iterate through the validation DataLoader
        for batch_idx, (x, _) in enumerate(val_loader):
            # Transfer the batch of validation data to the specified device
            x = x.to(DEVICE)
        
            # Perform a forward pass through the model to obtain reconstructions and latent codes
            x_tilde, z_e_x, z_q_x = model(x)

            # Compute the reconstruction loss (mean squared error) between reconstructions and inputs
            loss_recons = F.mse_loss(x_tilde, x)
            total_loss_recons += loss_recons.item()

            # Compute the VQ loss (mean squared error) between latent codes and quantized versions
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            total_loss_vq += loss_vq.item()

            # Append the reconstruction and VQ losses to the val_loss list
            val_loss.append(to_scalar([loss_recons, loss_vq]))

            # Compute SSIM for the current batch and accumulate it
            ssim_current = compute_ssim(x, x_tilde)
            ssim_accum += ssim_current
            
            # Update batch count
            batch_count += 1
            num_batches += 1
    
    # Calculate average reconstruction loss, VQ loss, and SSIM score over all batches
    if num_batches > 0:
        avg_loss_recons = total_loss_recons / num_batches
        avg_loss_vq = total_loss_vq / num_batches
        avg_ssim = ssim_accum / num_batches
    else:
        avg_loss_recons = 0
        avg_loss_vq = 0
        avg_ssim = 0
    
    # Append the epoch-level losses and SSIM score to respective lists
    val_losses.append((avg_loss_recons, avg_loss_vq))
    ssim_scores.append(avg_ssim)

    # Calculate a combined evaluation metric that balances SSIM and reconstruction loss
    combined_metric = ALPHA * avg_ssim - BETA * avg_loss_recons
        
    return combined_metric

"""
test(void)
----------
This function assesses the quality of the VAE model's reconstructions on a test dataset.
It computes the Structural Similarity Index (SSIM) score, a measure of structural similarity
between the original test data and the reconstructed data, for each batch of test data.

Input: None
Output: avg_ssim (float): The average SSIM score across all batches in the test dataset,
    representing the quality of reconstructions.
"""  
def test():
    model.eval()  # Switch to evaluation mode

    # Initialize variables for accumulating SSIM scores and batch count
    ssim_accum = 0.0  # Accumulator for SSIM scores
    batch_count = 0   # Counter for batches

    # Disable gradient computation during testing
    with torch.no_grad():  # No gradient required for testing
        # Iterate through the test DataLoader
        for batch_idx, (x, _) in enumerate(test_loader):
            # Transfer the batch of test data to the specified device
            x = x.to(DEVICE)

            # Perform a forward pass through the model to obtain reconstructions (ignore latent codes)
            x_tilde, _, _ = model(x)

            # Compute SSIM for the current batch and accumulate it
            ssim_accum += compute_ssim(x, x_tilde)

            # Update batch count
            batch_count += 1

    # Calculate the average SSIM for all batches
    avg_ssim = ssim_accum / batch_count

    # print average ssim score for the test set
    print(f"Average Test SSIM: {avg_ssim:.4f}")

    return avg_ssim  # return SSIM score

# Initialize empty list to store training losses per epoch
train_losses_epoch = []
# Initialize variable to store the total reconstruction loss
total_loss_recons = 0.0

if __name__ == '__main__':
    # Iterate over epochs
    for epoch in range(1, N_EPOCHS):
        print(f"Epoch {epoch}:")

        # Call the training function for the current epoch
        train()
       
        # Calculate the combined metric by evaluating the model on the validation dataset
        combined_metric = validate()

        # Check if the combined metric for this epoch is an improvement
        if combined_metric > BEST_METRIC:                        
            BEST_METRIC = combined_metric # Update the best metric
            BEST_EPOCH = epoch # Update the best epoch
            print("Saving model based on improved combined metric!")

            # Extract the dataset name from the path
            dataset_name = DATASET_PATH.split('/')[-1]  # Extracts the name "OASIS" from the path
            # Save the model's state dictionary to a file
            torch.save(model.state_dict(), f'samples6/checkpoint_epoch{BEST_EPOCH}_vqvae.pt')
            
            # Write the best epoch to a text file for reference
            with open("best_epoch.txt", "w") as file:
                file.write(str(BEST_EPOCH))

            # Log the best SSIM and reconstruction loss
            BEST_SSIM = ssim_scores[-1]
            BEST_RECONS_LOSS = val_losses[-1][0]
        
        else:
            print(f"Not saving model! Last best combined metric: {BEST_METRIC:.4f}, SSIM: {BEST_SSIM:.4f}, Reconstruction Loss: {BEST_RECONS_LOSS:.4f}")
        
        # Generate and save samples at the end of the first epoch for comparison
        if epoch == 1:
            generate_samples(model, test_loader, epoch) # prints first constructed image for comparison
        # Generate and save samples at the end of every 'save_interval' epoch
        if epoch % save_interval == 0:
            generate_samples(model, test_loader, epoch)
        
        # Step the learning rate scheduler to adjust the learning rate if needed
        scheduler.step()

    # Load the best model before testing
    model.load_state_dict(torch.load(MODEL_PATH_TEMPLATE.format(BEST_EPOCH)))
    # Test the model on the test dataset and calculate the average SSIM score
    test_ssim = test()
    print(f"Average SSIM on Test Set: {test_ssim:.4f}")

    # Plot training and validation losses and SSIM scores
    plot_losses_and_scores(train_losses_epoch, val_losses, ssim_scores)

    # Saving the best epoch for future reference (to be used in predict.py)
    with open("best_epoch.txt", "w") as file:
        file.write(str(BEST_EPOCH))
    
