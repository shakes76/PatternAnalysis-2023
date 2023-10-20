import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import DDPM_UNet, UNet
from dataset import ADNIDataset

from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ensure the size is consistent
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1] for better training stability
])

dataset = ADNIDataset(root_dir="./AD_NC",  train=True, transform=transform)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


NUM_EPOCHS=30
LEARNING_RATE=0.001
NUM_STEPS=1000
MIN_BETA=1e-3
MAX_BETA=0.03 
DEVICE= torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DDPM_UNet(UNet(num_steps=NUM_STEPS), num_steps=NUM_STEPS, min_beta=MIN_BETA, max_beta=MAX_BETA, device=DEVICE)

def training_loop(model, loader, num_epochs, optimizer, device, lr_scheduler=None, gradient_clip=None, store_path="model_model.pt"):
    mean_squared_error = nn.MSELoss()
    best_loss = float("inf")
    num_steps = model.num_steps

    for epoch in tqdm(range(num_epochs), desc="Training progress", colour="#00ff00"):
        epoch_loss = 0.0
        for step, batch in enumerate(loader):  # Remove tqdm from here
            loss = compute_loss(batch, model, mean_squared_error, device, num_steps)

            optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            if gradient_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            epoch_loss += loss.item() * len(batch) / len(loader.dataset)

            # Logging within the epoch
            if step % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Step {step}/{len(loader)} - Loss: {loss.item():.3f}")

        if lr_scheduler:
            lr_scheduler.step()

        save_model_if_best(epoch_loss, model, best_loss, store_path)
        print(f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}")

def compute_loss(batch, model, mse, device, num_steps):
    original_image = batch.to(device)
    num_samples = len(original_image)

    # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
    noise = torch.randn_like(original_image).to(device)
    time_step = torch.randint(0, num_steps, (num_samples,)).to(device)

    # Computing the noisy image based on the original image and the time-step (forward process)
    noisy_images = model(original_image, time_step, noise)

    # Getting model estimation of noise based on the images and the time-step
    estimated_noise = model.denoise(noisy_images, time_step.reshape(num_samples, -1))

    return mse(estimated_noise, noise)

def save_model_if_best(epoch_loss, model, best_loss, store_path):
    if best_loss > epoch_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), store_path)
        print(f"New best model saved at epoch loss: {epoch_loss:.3f}")

