# CONSTANTS AND HYPERPARAMETERS:

from modules import VQVAE, device


LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 40 # realistically stopped earlier by the validation set
CODEBOOK_SIZE = 512

# Weights for the loss functions
L2_WEIGHT = 0.05
SSIM_WEIGHT = 1

# Constants for early stopping
PATIENCE = 12
best_val_loss = float('inf')
counter = 0







def main():
    model = model = VQVAE(CODEBOOK_SIZE).to(device)