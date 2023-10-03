import os
import dataset
import module
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch import nn
from torch.optim import Adam

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory to save images
output_dir = './image_output'

# Number of Epochs for training
epochs = 100

# Create a model instance from module.py
model = module.UNet()

# Adam Optimizer for training the model
optimizer = Adam(model.parameters(), lr=0.001)

## Code referenced from:
# https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL
# https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb

# Loss function
def get_loss(model, x_0, t):
    """
    Loss function using L1 loss (Mean Absolute Error)
    L_t (for random time step t given noise ~ N(0, I)):
    L_simple = E_(t,x_0,e)[||e - e_theta(x_t, t)||^2]
    where e is added noise, e_theta is predicted noise
    x_0: image
    """
    x_noise, noise = module.forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noise, t)
    return F.l1_loss(noise, noise_pred)

# Sampling
@torch.no_grad()
def sample_timestep(x, t):
    """
    Noise in the image x is predicted and returns denoised image
    """
    betas_t = module.get_index_from_list(module.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = module.get_index_from_list(
        module.sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = module.get_index_from_list(module.sqrt_recip_alphas, t, x.shape)
    
    # call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = module.get_index_from_list(module.posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    

def reverse_transform_image(image, epoch, output_dir):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])
    
    # save image here
    image_path = os.path.join(output_dir, f'epoch_{epoch:03d}_generated.png')
    vutils.save_image(image, image_path)

@torch.no_grad()
def sample_save_image(epoch, output_dir):
    # Sample noise
    img_size = 64
    img = torch.randn((1, 1, img_size, img_size), device=device)
    num_images=10
    stepsize = int(module.T/num_images)
    
    for i in range(0, module.T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t)
        # Maintain natural range of distribution
        img = torch.clamp(img, -1, 1)
        if i % stepsize == 0:
            reverse_transform_image(img.detach().cpu(), epoch, output_dir)


best_loss = float('inf')  # Initialize with a high value
best_model_state_dict = None  # Variable to store the state_dict of the best model
validate_every_n_epochs = 5 # Variable to validate state of model every 5 epochs

for epoch in range(epochs):
    
    # Training Loop
    for step, batch_image in enumerate(dataset.train_loader):
        image, label = batch_image
        optimizer.zero_grad()
        print(image.shape)
        batch_size = image.shape[0]
        batch = image.to(device)
        
        t = torch.randint(0, module.T, (batch_size,), device=device).long()
        loss = get_loss(model, batch[0], t)
        loss.backward()
        optimizer.step() 
        
        if epoch % 10 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()}")
            sample_save_image(epoch, output_dir)
            
    # Validation loop (Written by ChatGPT3.5)
    if epoch % validate_every_n_epochs == 0:
        model.eval()  # Set the model to evaluation mode
        total_validation_loss = 0.0
        total_validation_samples = 0
        
        with torch.no_grad():
            for batch_image in dataset.validation_loader:
                image, label = batch_image
                batch_size = image.shape[0]
                batch = image.to(device)
                t = torch.randint(0, module.T, (batch_size,), device=device).long()
                validation_loss = get_loss(model, batch[0], t)
                total_validation_loss += validation_loss.item() * batch_size
                total_validation_samples += batch_size

        # Calculate average validation loss
        average_validation_loss = total_validation_loss / total_validation_samples

        # Check if the current model has a lower validation loss than the best so far
        if average_validation_loss < best_loss:
            best_loss = average_validation_loss  # Update the best loss
            best_model_state_dict = model.state_dict()  # Save the state_dict of the best model

            torch.save(best_model_state_dict, 'best_model.pth')
        
        # Print validation results
        print(f"Epoch {epoch} | Validation Loss: {average_validation_loss}")
