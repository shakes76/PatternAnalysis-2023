import torch
from torch import nn, optim
from dataset import load_dataset
from modules import Generator, Discriminator
from math import log2
from torchvision.utils import save_image
import os



"""
Contains the source code for training, validating, testing and saving the model.
The model should be imported from "modules.py" and the data loader should be imported from "dataset.py".
Make sure to plot the losses and metrics during training.
"""

#The following code is modified from
#https://www.kaggle.com/code/tauilabdelilah/stylegan-implementation-from-scratch-pytorch

EXPORT_PATH = "saved_examples"
START_TRAIN_AT_IMG_SIZE = 8
DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE           = 1e-3
BATCH_SIZES             = [256, 128, 64, 32, 16, 8, 4] #TODO try increasing batch sizes #TODO this batch sizes isnt used for anything except loading??
CHANNELS_IMG            = 1
Z_DIM                   = 256
W_DIM                   = 256
IN_CHANNELS             = 256
LAMBDA_GP               = 10
PROGRESSIVE_EPOCHS      = [20] * len(BATCH_SIZES)

#Factors for progressive growing
factors = [1, 1, 1, 1, 1/2, 1/4, 1/8, 1/16, 1/32]



def save_generator(gen, steps):
    if not os.path.exists(f"{EXPORT_PATH}/step{steps}"):
        os.makedirs(f"{EXPORT_PATH}/step{steps}")
    torch.save(gen.state_dict(), f"{EXPORT_PATH}/step{steps}/generator{steps}.pt")


def gradient_penalty(discriminator, real, fake, alpha, train_step, device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate discriminator scores
    mixed_scores = discriminator(interpolated_images, alpha, train_step)
 
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



gen_losses = []
discriminator_losses = []

#Training function-----------------------------------------------
def train_fn(
    discriminator,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_discriminator,
    opt_gen,
):

    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(DEVICE)
        cur_batch_size = real.shape[0]


        noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)

        fake = gen(noise, alpha, step)
        discriminator_real = discriminator(real, alpha, step)
        discriminator_fake = discriminator(fake.detach(), alpha, step)
        gp = gradient_penalty(discriminator, real, fake, alpha, step, device=DEVICE)
        loss_discriminator = ( #TODO what??
            -(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
            + LAMBDA_GP * gp
            + (0.001 * torch.mean(discriminator_real ** 2))
        )

        discriminator.zero_grad()
        loss_discriminator.backward()
        opt_discriminator.step()

        gen_fake = discriminator(fake, alpha, step)
        loss_gen = -torch.mean(gen_fake)

        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        # Update alpha and ensure less than 1
        alpha += cur_batch_size / (
            (PROGRESSIVE_EPOCHS[step] * 0.5) * len(dataset)
        )
        alpha = min(alpha, 1)

        loss_g = loss_discriminator.item()
        loss_d = loss_gen.item()
        gen_losses.append(loss_g)
        discriminator_losses.append(loss_d)
    
        gp = gp.item(),
        loss_discriminator = loss_d,
        loss_gen = loss_g

    return alpha
#----------------------------------------------------------------



#Training--------------------------------------------------------
gen = Generator(
        Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG
    ).to(DEVICE)
discriminator = Discriminator(IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
# initialize optimizers
opt_gen = optim.Adam([{"params": [param for name, param in gen.named_parameters() if "map" not in name]},
                        {"params": gen.map.parameters(), "lr": 1e-5}], lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_discriminator = optim.Adam(
    discriminator.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99)
)


gen.train()
discriminator.train()

# start at step that corresponds to img size that we set in config
step = int(log2(START_TRAIN_AT_IMG_SIZE / 4))
for num_epochs in PROGRESSIVE_EPOCHS[step:]:
    alpha = 1e-5   # start with very low alpha
    loader, dataset = load_dataset(4 * 2 ** step, BATCH_SIZES[step])  
    print(f"Current image size: {4 * 2 ** step}")

    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        alpha = train_fn(
            discriminator,
            gen,
            loader,
            dataset,
            step,
            alpha,
            opt_discriminator,
            opt_gen
        )

    gen.train()
    save_generator(gen, step)
    step += 1  # progress to the next img size



import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(gen_losses,label="G")
plt.plot(discriminator_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
#----------------------------------------------------------------