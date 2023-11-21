from imports import *
from dataset import process_dataset
from modules import *
from utils import *

def train_diffusion_network(model, optimizer, epochs, batch_size, save_path=None, plot_path=None):
    # Initialize DataLoader and other variables
    train_dataloader = process_dataset(batch_size=batch_size, is_validation=False, pin_memory=True)
    
    # Wrap model for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model) 
    model = model.to(device)
    
    losses = []
    
    model.train()
    # Training loop
    for epoch in range(epochs):
        for step, batch in enumerate(train_dataloader):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Generate random time indices for each batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # Backpropagation
            loss = get_loss(model, batch, t)
            loss.backward()
            optimizer.step()
            
            # Collect loss from all GPUs
            if torch.cuda.device_count() > 1:
                loss = loss.mean()
                
            losses.append(loss.item())
            
            # Log the loss value every 100 iterations
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Iteration {step+1}, Loss: {loss.item()}")

    # Save the trained model
    if save_path:
        if torch.cuda.device_count() > 1:
            # Save the state_dict of the inner model
            torch.save(model.module.state_dict(), save_path) 
        else:
            torch.save(model.state_dict(), save_path)

    # Plot and save the training loss
    if plot_path:
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(plot_path)