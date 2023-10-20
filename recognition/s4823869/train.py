import torch
from torch import optim
import os
import matplotlib.pyplot as plt
import random
import numpy as np

import modules
import dataset

torch.mps.empty_cache()
# Choose device ('mps' for Memory Persistence Service, 'cpu' if unavailable)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Constants
Z_DIM = 512
W_DIM = 512
LAMBDA_GP = 10
BATCH_SIZES = [256, 128, 64, 32, 16, 8]
PROGRESSIVE_EPOCHS = [1] * (len(BATCH_SIZES)-5)
IN_CHANNELS = 512
CHANNELS_IMG = 3
LR_GEN = 1e-3
LR_CRITIC = 5e-4
START_TRAIN_IMG_SIZE = 4

gen_losses = []
critic_losses = []

# Function to calculate gradient penalty
def calculate_gradient_penalty(critic, real, fake, alpha, step, device):
    batch_size, _, _, _ = real.shape
    beta = torch.rand((batch_size, 1, 1, 1)).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    mixed_scores = critic(interpolated_images, alpha, step)


    gradients = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_norm = gradients.view(gradients.shape[0], -1).norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

# Train function
def train(critic, gen, loader, step, alpha, opt_critic, opt_gen):
    for batch_idx, real in enumerate(loader):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)
        fake = gen(noise, alpha, step)

        critic_real = critic(real, alpha, step)  # Pass step and alpha to the discriminator
        critic_fake = critic(fake.detach(), alpha, step)  # Pass step and alpha to the discriminator
        gp = calculate_gradient_penalty(critic, real, fake, alpha, step, DEVICE)
        # Pass step to the gradient penalty function

        loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
                + (0.001) * torch.mean(critic_real ** 2)
        )
        critic_losses.append(abs(loss_critic.item()))
        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, alpha, step)  # Pass step and alpha to the discriminator
        loss_gen = -torch.mean(gen_fake)
        gen_losses.append(abs(loss_gen.item()))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        alpha += cur_batch_size / (PROGRESSIVE_EPOCHS[step] * 0.5 * len(loader.dataset))
        alpha = min(alpha, 1)

        print(
            f"Step [{step + 1}/{len(PROGRESSIVE_EPOCHS)}] Batch [{batch_idx + 1}/{len(loader)}] loss_gen: {loss_gen:.4f}")

    return alpha


# Model initialization
# Model initialization
gen = modules.Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
critic = modules.Discriminator(IN_CHANNELS, CHANNELS_IMG).to(DEVICE)

# Optimizers
gen_params = [{'params': [param for name, param in gen.named_parameters() if 'map' not in name]}]
gen_params += [{'params': gen.map.parameters(), 'lr': 1e-5}]
opt_gen = optim.Adam(gen_params, lr=LR_GEN, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC, betas=(0.0, 0.99))

# Rest of the code (training loop and progressive training) remains the same


# Set training mode
gen.train()
critic.train()

# Progressive training
step = int(np.log2(START_TRAIN_IMG_SIZE / 4))
epoch_losses = []

for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-7
    loader, data = dataset.get_loader(4 * 2 ** step)
    print('Present image size: ' + str(4 * 2 ** step))

    for epoch in range(num_epochs):
        train(critic, gen, loader, step, alpha, opt_critic, opt_gen)

    epoch_loss_gen = sum(gen_losses) / len(gen_losses)
    epoch_losses.append(epoch_loss_gen)

    step += 1
    gen_losses = []

# Save the models
torch.save(gen.state_dict(), 'Generator.pth')

# Plot the losses of generator and critic
def plot_losses(epoch_losses, critic_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label="Generator Loss per Epoch", color="yellow")
    plt.plot(critic_losses, label="Critic Loss per Batch", color="green")
    plt.title("Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.show()
save_path = os.path.join("output_images", "losses_plot.png")
plot_losses(epoch_losses, critic_losses)
