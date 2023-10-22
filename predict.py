"""
predict.py

Description:
    This module provides functions to generate GIFs of new images from a given generative model. 
    The main function, create_gif, processes the noise tensor with the model across multiple steps 
    and determines the frame indices for the GIF. The generated frames are then normalized and stored 
    in the desired GIF format.

Author:
    Santiago Rodrigues (46423232)
"""


import numpy as np
import torch
import imageio

def create_gif(model, num_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=1, h=224, w=224):
    """Generate a GIF of new images from the given model.
    
    Args:
    - model: The generative model.
    - num_samples: Number of images to generate.
    - device: Computation device (GPU/CPU).
    - frames_per_gif: Number of frames in the GIF.
    - gif_name: Name of the output GIF.
    - c, h, w: Channels, height, and width of the images.
    
    Returns:
    - Tensor of generated images.
    """
    frame_idxs = get_frame_indices(model, frames_per_gif)
    frames = []

    with torch.no_grad():
        device = device if device else model.device
        x = torch.randn(num_samples, c, h, w).to(device)

        for _, t in enumerate(reversed(list(range(model.num_steps)))):
            x = process_noise_with_model(x, model, t, num_samples, device, c, h, w)
            if t in frame_idxs or t == 0:
                frames.append(create_frame(x, num_samples))

    store_gif(frames, gif_name, frames_per_gif)
    return x

def get_frame_indices(model, frames_per_gif):
    """Determine the indices for frames in the GIF."""
    return np.linspace(0, model.num_steps, frames_per_gif).astype(np.uint)

def process_noise_with_model(x, model, t, num_samples, device, c, h, w):
    """Process the noise tensor with the model for a single step."""
    time_tensor = (torch.ones(num_samples, 1) * t).to(device).long()
    eta_theta = model.denoise(x, time_tensor)

    alpha_t = model.alphas[t]
    alpha_t_bar = model.alpha_bars[t]

    x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

    if t > 0:
        z = torch.randn(num_samples, c, h, w).to(device)
        beta_t = model.betas[t]
        sigma_t = beta_t.sqrt()
        x = x + sigma_t * z

    return x

def create_frame(x, num_samples):
    """Normalize and arrange images into a frame for the GIF."""
    normalized = x.clone()
    for i in range(len(normalized)):
        normalized[i] -= torch.min(normalized[i])
        normalized[i] *= 255 / torch.max(normalized[i])
        
    b1 = int(num_samples ** 0.5)
    frame = torch.cat(torch.chunk(torch.cat(torch.chunk(normalized, b1, dim=0), dim=2), b1, dim=0), dim=3)
    
    frame = frame.cpu().numpy().astype(np.uint8)
    return np.repeat(frame, 3, axis=-1)

def store_gif(frames, gif_name, frames_per_gif):
    """Store the frames as a GIF."""
    with imageio.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            writer.append_data(frame)
            if idx == len(frames) - 1:
                for _ in range(frames_per_gif // 3):
                    writer.append_data(frames[-1])
