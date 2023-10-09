from modules import DiffusionProcess, DiffusionNetwork
from dataset import process_dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import torch

# Initialize models and dataset
betas = [0.1] * 100  # Example list of betas for 100 steps
num_steps = 100
diffusion_process = DiffusionProcess(betas, num_steps)
diffusion_network = DiffusionNetwork()

optimizer = optim.Adam(diffusion_network.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()  # Example loss function

batch_size = 8
dataloader = process_dataset(batch_size=batch_size)

# Training loop
losses = []
for epoch in range(10):  # Example: 10 epochs
    for i, batch in enumerate(dataloader):
        # Apply diffusion process
        diffused_batch = diffusion_process(batch)
        
        # Generate image from diffused image
        output = diffusion_network(diffused_batch)
        
        # Compute loss
        loss = criterion(output, batch)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        print(f"Epoch {epoch+1}, Iteration {i+1}, Loss: {loss.item()}")

# Plotting loss
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
