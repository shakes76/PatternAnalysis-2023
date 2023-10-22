"""
train.py

Description:
    This module provides functions and configurations to train the DDPM_UNet model
    on the ADNIDataset. It sets up dataset transformations, loading, model initialization, 
    and defines the training loop with functionalities such as computing loss and model saving.
    The main focus is to train the model for denoising purposes.

Author:
    Santiago Rodrigues (46423232)
"""


import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import DDPM_UNet, UNet
from dataset import ADNIDataset

from tqdm import tqdm

# Define transformations for preprocessing the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to ensure consistent size
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize((0.5,), (0.5,)),  # Normalize images to range [-1, 1] for improved training stability
])

# Load the dataset from the root directory with the given transformations
dataset = ADNIDataset(root_dir="./AD_NC",  train=True, transform=transform)
# Create a DataLoader for batching and shuffling the dataset
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define hyperparameters for training
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
NUM_STEPS = 1000
MIN_BETA = 1e-3
MAX_BETA = 0.03 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with the specified architecture and hyperparameters
model = DDPM_UNet(UNet(num_steps=NUM_STEPS), num_steps=NUM_STEPS, min_beta=MIN_BETA, max_beta=MAX_BETA, device=DEVICE)

def training_loop(model, loader, num_epochs, optimizer, device, lr_scheduler=None, gradient_clip=None, store_path="model_model.pt"):
    """
    Train the model using the given dataset loader.

    Parameters:
    - model: The model to be trained.
    - loader: DataLoader for the training dataset.
    - num_epochs: Number of epochs to train.
    - optimizer: Optimizer to use for training.
    - device: Device to run training on.
    - lr_scheduler (optional): Learning rate scheduler.
    - gradient_clip (optional): Value for gradient clipping.
    - store_path (optional): Path to store the best model.
    """
    mean_squared_error = nn.MSELoss()
    best_loss = float("inf")
    num_steps = model.num_steps

    # Training loop
    for epoch in tqdm(range(num_epochs), desc="Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):
            loss = compute_loss(batch, model, mean_squared_error, device, num_steps)
            
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping if specified
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()
            epoch_loss += loss.item() * len(batch) / len(loader.dataset)

            # Log training progress within the epoch
            if step % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {step}/{len(loader)} - Loss: {loss.item():.3f}")

        # Update learning rate if a scheduler is provided
        if lr_scheduler:
            lr_scheduler.step()

        # Save the best model based on epoch loss
        save_model_if_best(epoch_loss, model, best_loss, store_path)
        print(f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}")

def compute_loss(batch, model, mse, device, num_steps):
    """
    Compute the loss for a batch of images.

    Parameters:
    - batch: Batch of images to compute the loss for.
    - model: Model to use for computing the loss.
    - mse: Mean squared error loss function.
    - device: Device to run computations on.
    - num_steps: Number of steps in the model.

    Returns:
    - loss: Computed loss for the batch.
    """
    original_image = batch.to(device)
    num_samples = len(original_image)

    # Generate noise for each image in the batch
    noise = torch.randn_like(original_image).to(device)
    time_step = torch.randint(0, num_steps, (num_samples,)).to(device)

    # Use the model to compute noisy images based on the original images and time-step
    noisy_images = model(original_image, time_step, noise)

    # Estimate the noise based on the noisy images and time-step
    estimated_noise = model.denoise(noisy_images, time_step.reshape(num_samples, -1))

    return mse(estimated_noise, noise)

def save_model_if_best(epoch_loss, model, best_loss, store_path):
    """
    Save the model if the current epoch loss is better than the best known loss.

    Parameters:
    - epoch_loss: Loss of the current epoch.
    - model: Model to save.
    - best_loss: Best known loss so far.
    - store_path: Path to store the best model.
    """
    if best_loss > epoch_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), store_path)
        print(f"New best model saved at epoch loss: {epoch_loss:.3f}")
