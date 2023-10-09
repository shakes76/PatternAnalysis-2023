import torch
from torch import optim
from tqdm import tqdm
import os
from math import log2

import modules
import dataset

'''
This file progressively trains the styleGAN model using gradient penalty regularization on the discriminator as
well as the typical adversarial training framework;
These techniques are referenced from:
https://www.kaggle.com/code/tauilabdelilah/stylegan-implementation-from-scratch-pytorch
'''

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Z_DIm = 512
W_DIM = 512
LAMBDA_GP = 10 # coefficient of gradient penalty
BATCH_SIZES = [256,256,128,64,32,16]
PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
IN_CHANNELS = 512
CHANNELS_IMG = 3
LR = 1e-3
START_TRAIN_IMG_SIZE = 4

# Regularization on the discriminator / critic
def gradient_penalty(critic, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images, alpha, train_step)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def train_fn(critic, gen, loader, dataset, step, alpha, opt_critic, opt_gen):
    loop = tqdm(loader, leave=True)

    for batch_idx, real in enumerate(loop):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]
        noise = torch.randn(cur_batch_size, Z_DIm).to(DEVICE) # z
        fake  = gen(noise, alpha, step)
        critic_real = critic(real, alpha, step)
        critic_fake = critic(fake.detach(), alpha, step)
        gp = gradient_penalty(critic, real, fake, alpha, step, DEVICE)
        # enhanced version of the discriminator loss
        loss_critic = (
            -(torch.mean(critic_real) - torch.mean(critic_fake))
            + LAMBDA_GP * gp
            + (0.001) * torch.mean(critic_real ** 2)
        )

        critic.zero_grad()
        loss_critic.backward()
        opt_critic.step()

        gen_fake = critic(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        alpha += cur_batch_size / (
            PROGRESSIVE_EPOCHS[step] * 0.5 * len(dataset)
        )
        alpha = min(alpha,1)


        loop.set_postfix(
            gp = gp.item(),
            loss_critic = loss_critic.item(),
            loss_gen = loss_gen.item() # display both losses during training
        )
    return alpha

# Model initialization
gen = modules.Generator(Z_DIm, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
critic = modules.Discriminator(IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
# Optimization
opt_gen = optim.Adam([{'params': [param for name, param in gen.named_parameters() if 'map' not in name]},
                     {'params': gen.map.parameters(), 'lr': 1e-5}], lr=LR, betas =(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr= LR, betas =(0.0, 0.99))

# train mode
gen.train()
critic.train()

# Progressive training
step = int(log2(START_TRAIN_IMG_SIZE / 4))
for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-7
    loader, data = dataset.get_loader(4*2**step)
    print('Current image size: '+str(4*2**step))

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/ {num_epochs}')
        alpha = train_fn(critic, gen, loader, data, step, alpha, opt_critic, opt_gen)
    #generate_examples(gen, step)
    step +=1

# Save the models
torch.save(gen.state_dict(), 'OASIS_style_gan_generater.pth')
#torch.save(critic.state_dict(), 'OASIS_style_gan_critic.pth')

