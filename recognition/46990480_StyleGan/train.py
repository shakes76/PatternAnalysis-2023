"""
Contains the training and validation steps for each of the models
"""
import torch
from torch import optim
from dataset import generateDataLoader
from modules import Generator, Discriminator, MappingNetwork, PathLengthPenalty
from tqdm import tqdm
import time
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path Parameters
modelName = "256_train_Wpre_256_lat"
# data_path_root = "./data/OASIS/" # Rangpur
data_path_root = "/Datasets/keras_png_slices_data/" # Desktop
output_path = "./"
save_path = "./Models/"

# Training Hyper Parameters
num_epochs = 2
learning_rate = 0.0001
channels = 3  # Number of channels (rgb images, so we have 3 channels in this case)
batch_size = 64  # Training Batch Size
image_size = 64  # Spatial size of the training images - OASIS 256px
log_resolution = 8 # for 256*256 image size
image_height = 2**log_resolution
image_width = 2**log_resolution
z_dim = 256  # in the original paper these were initialised to 512
w_dim = 256  # style vector dimension
lambda_gp = 10  # Lambda value for the gradient penalty

# ----------------
# Data
print("> Loading Dataset")
trainset, train_loader, *otherLoaders = generateDataLoader(image_height, image_width, batch_size, data_path_root)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)  # Data Loader
total_steps = len(train_loader)
print("> Dataset Ready")

# ----------------
# Models
print("> Configuring Models")
print("> Training Model: " + modelName)
generator = Generator(log_resolution, w_dim).to(device)
discriminator = Discriminator(log_resolution).to(device)
mapping_network = MappingNetwork(z_dim, w_dim).to(device)
path_length_penalty = PathLengthPenalty(0.99).to(device)
opt_generator = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_discriminator = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=learning_rate, betas=(0.0, 0.99))

# Generator & Discriminator info
print("Generator No. of Parameters:", sum([param.nelement() for param in generator.parameters()]))
print(generator)
print("Discriminator No. of Parameters:", sum([param.nelement() for param in discriminator.parameters()]))
print(discriminator)
print("Mapping Network No. of Parameters:", sum([param.nelement() for param in mapping_network.parameters()]))
print(mapping_network)

# ----------------
# Training the model
# Put the models into training mode
generator.train()
discriminator.train()
mapping_network.train()

def WGAN_GP_LOSS(discriminator, real, fake, device="cpu"):
    '''
    Computes gradient penalty (loss) for WGAN-GP
    '''
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate discriminator scores
    mixed_scores = discriminator(interpolated_images)
 
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

def get_w(batch_size, log_resolution):
    '''
    Creates a style latent vector w, from a random noise z latent vector.
    '''
    # Random noise z latent vector
    z = torch.randn(batch_size, w_dim).to(device)

    # Forward pass z through the mapping network to generate w latent vector
    w = mapping_network(z)
    return w[None, :, :].expand(log_resolution, -1, -1)

def get_noise(batch_size):
    '''
    Generates a random noise vector for a batch of images
    '''
    noise = []
    resolution = 4

    for i in range(log_resolution):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        noise.append((n1, n2))

        resolution *= 2

    return noise

def generate_examples(gen, epoch, n=100):
    '''
    Saves n number of sample images generated from random noise at a specified training epoch.
    '''
    gen.eval()
    for i in range(n):
        with torch.no_grad():
            w = get_w(1, log_resolution)
            noise = get_noise(1)
            img = gen(w, noise)
            if not os.path.exists(f'saved_examples_{modelName}/epoch{epoch}'):
                os.makedirs(f'saved_examples_{modelName}/epoch{epoch}')
            save_image(img*0.5+0.5, f"saved_examples_{modelName}/epoch{epoch}/img_{i}.png")

    gen.train()

# Lists to keep track of progress
G_losses = []
D_losses = []

def train_fn(
    discriminator,
    generator,
    path_length_penalty,
    loader,
    opt_discriminator,
    opt_generator,
    opt_mapping_network,
):
    '''
    Main training loop for the model
    '''
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop): # load a batch of images (depicted by the batch size)
        # Real image batch
        real = real.to(device)

        # Batch size
        current_batch_size = real.shape[0]

        w = get_w(current_batch_size, log_resolution)
        noise = get_noise(current_batch_size)
        with torch.cuda.amp.autocast():
            # Generate a fake image batch with the Generator
            fake = generator(w, noise)
            
            # Forward pass the fake image through the discriminator
            discriminator_fake = discriminator(fake.detach())
            
            # Forward pass the real image through the discriminator
            discriminator_real = discriminator(real)
            criterion = WGAN_GP_LOSS(discriminator, real, fake, device=device)
            loss_discriminator = (
                -(torch.mean(discriminator_real) - torch.mean(discriminator_fake))
                + lambda_gp * criterion
                + (0.001 * torch.mean(discriminator_real ** 2))
            )

        # Update Discriminator Neural Network => maximize log(D(x)) + log(1 - D(G(z)))
        discriminator.zero_grad()
        loss_discriminator.backward()
        opt_discriminator.step()

        # Forward pass the fake batch of generated images through the discriminator
        gen_fake = discriminator(fake)

        # Compute the generator loss
        loss_gen = -torch.mean(gen_fake)

        if batch_idx % 16 == 0:
            plp = path_length_penalty(w, fake)
            if not torch.isnan(plp):
                loss_gen = loss_gen + plp

        # Update the networks
        mapping_network.zero_grad()
        generator.zero_grad()
        loss_gen.backward()
        opt_generator.step()
        opt_mapping_network.step()
        G_losses.append(gp.item())
        D_losses.append(loss_discriminator.item())

        # logging with tqdm
        loop.set_postfix(
            gp=criterion.item(),
            loss_critic=loss_discriminator.item(),
        )

print("> Training")
start = time.time()  # time generation

# Train loop
for epoch in range(num_epochs):
    train_fn(
        discriminator,
        generator,
        path_length_penalty,
        train_loader,
        opt_discriminator,
        opt_generator,
        opt_mapping_network,
    )

    # Save example images every 50 epochs
    if epoch % 50 == 0:
    	generate_examples(generator, epoch)

# Save the models
torch.save(generator.state_dict(), save_path + f"GENERATOR_{modelName}.pth")
torch.save(discriminator.state_dict(), save_path + f"DISCRIMINATOR_{modelName}.pth")
torch.save(mapping_network.state_dict(), save_path + f"MAPPING_NETWORK_{modelName}.pth")

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed / 60) + " mins in total")

# Generate & Save Training Loss plot
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="Generator")
plt.plot(D_losses, label="Discriminator")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
plt.savefig(f'{output_path}saved_examples_{modelName}/trainingLossPlot.png')

print("> Done")
