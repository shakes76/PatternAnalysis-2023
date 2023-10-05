# Import necessary modules and functions from the 'train' module 
from train import * 
# Import other necessary libraries
from torch import optim
from torchvision.utils import save_image
import os
import matplotlib.pyplot as plt


# Function to generate example images using a generator model
def generate_examples(gen, epoch, n=20):
    # Set the generator model in evaluation mode
    gen.eval()
    
    # Generate 'n' example images
    for i in range(n):
        with torch.no_grad():
            w_values = [1, 2, 3, 4, 5]
            for value in w_values:
                # Generate random latent vector 'w'
                w = get_w(value)
                
                # Generate random noise
                noise = get_noise(1)
                
                # Generate an image using the generator model
                img = gen(w, noise)
                
                # Create a directory to save the images for the current epoch if it doesn't exist
                if not os.path.exists(f'generated_images/epoch{epoch}/w{value}'):
                    os.makedirs(f'generated_images/epoch{epoch}/w{value}')
                
                # Save the generated image with appropriate scaling
                save_image(img*0.5+0.5, f"generated_images/epoch{epoch}/w{value}/img_{i}.png")

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

Total_G_Losses = []
Total_D_Losses = []
# Training loop over a specified number of epochs
for epoch in range(EPOCHS):
    # Call the training function for the critic, generator, and mapping network
    G_Losses , D_Losses = train_fn(
        critic,
        gen,
        path_length_penalty,
        loader,
        opt_critic,
        opt_gen,
        opt_mapping_network,
    )
    
    Total_G_Losses.extend(G_Losses)
    Total_D_Losses.extend(D_Losses)
    
    # Generate example images for the current epoch
    generate_examples(gen, epoch)
    
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(Total_G_Losses,label="G")
plt.plot(Total_D_Losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('model_loss.png')
