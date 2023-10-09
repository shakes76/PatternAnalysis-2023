from modules import DiffusionProcess, DiffusionNetwork
from dataset import process_dataset
import torch
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model_path = "diffusion_network.pth"
diffusion_network = DiffusionNetwork()
diffusion_network.load_state_dict(torch.load(model_path))
diffusion_network.to(device)
diffusion_network.eval()  # set to evaluation mode

# Load data
batch_size = 9  # set batch size to 9 to generate 9 images
dataloader = process_dataset(batch_size=batch_size, is_validation=True)  # replace with your validation directory

# Prediction and Visualization
for i, batch in enumerate(dataloader):
    # Move batch to device
    batch = batch.to(device)
    
    # Generate image from input image
    with torch.no_grad():
        output = diffusion_network(batch)

    # Convert to numpy arrays
    output = output.cpu().numpy()
    
    # Plot 3x3 grid
    plt.figure(figsize=(10, 10))
    
    for j in range(9):
        plt.subplot(3, 3, j+1)
        plt.title(f"Image {j+1}")
        plt.imshow(output[j, 0], cmap='gray')
        plt.axis('off')
        
    plt.show()
    
    # Stop after first batch for demonstration
    break