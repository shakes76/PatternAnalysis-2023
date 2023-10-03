# Import necessary modules and functions from the 'train' module 
from train import * 
# Import other necessary libraries
from torch import optim
from torchvision.utils import save_image
import os

# Function to generate example images using a generator model
def generate_examples(gen, epoch, n=100):
    # Set the generator model in evaluation mode
    gen.eval()
    
    # Generate 'n' example images
    for i in range(n):
        with torch.no_grad():
            # Generate random latent vector 'w'
            w = get_w(1)
            
            # Generate random noise
            noise = get_noise(1)
            
            # Generate an image using the generator model
            img = gen(w, noise)
            
            # Create a directory to save the images for the current epoch if it doesn't exist
            if not os.path.exists(f'saved_examples/epoch{epoch}'):
                os.makedirs(f'saved_examples/epoch{epoch}')
            
            # Save the generated image with appropriate scaling
            save_image(img*0.5+0.5, f"saved_examples/epoch{epoch}/img_{i}.png")

    # Set the generator model back to training mode
    gen.train()

# Set the generator, critic, and mapping network models to training mode
gen.train()
critic.train()
mapping_network.train()

# Get a data loader for the specified dataset, resolution, and batch size
loader = get_loader(DATASET, LOG_RESOLUTION, BATCH_SIZE)

# Create a Path Length Penalty module with a specified decay rate and move it to the appropriate device
path_length_penalty = PathLengthPenalty(0.99).to(DEVICE)

# Initialize optimizers for the generator, critic, and mapping network with specified learning rates and betas
opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))
opt_mapping_network = optim.Adam(mapping_network.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.99))

# Training loop over a specified number of epochs
for epoch in range(EPOCHS):
    # Call the training function for the critic, generator, and mapping network
    train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )
    
    # Generate example images for the current epoch
    generate_examples(gen, epoch)
