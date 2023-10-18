from imports import *
from dataset import *
from modules import *
from utils import *

model = DiffusionNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 51
batch_size = 2
train_dataloader = process_dataset(batch_size=batch_size, is_validation=False)

losses = []
model.train()
for epoch in range(epochs):
    for step, batch in enumerate(train_dataloader):
        batch = batch.to(device)
        optimizer.zero_grad()

        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # Backpropagation
        loss = get_loss(model, batch, t)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Iteration {step+1}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), f"diffusion_network{epochs}.pth")

# Plotting loss
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')

# Save the plot
save_dir = os.path.expanduser("~/demo_eiji/sd/plots")
full_path = os.path.join(save_dir, "training_loss.png")
plt.savefig(full_path)