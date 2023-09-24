"""
Training, validating, testing and saving the model
"""
import torch.nn as nn
import torch.optim as optim
import time
import sys
from config import *
from dataset import *
from modules import SuperResolutionModel
from predict import save_model_output

print("PyTorch Version:", torch.__version__)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
sys.stdout.flush()

train_loader = get_train_dataloader()
# Don't shuffle test so that the model output is generated from a fixed sample
test_loader = get_test_dataloader(shuffle=False)

model = SuperResolutionModel(upscale_factor=dimension_reduce_factor).to(device)

# Define the loss function (MSE) and optimizer (Adam)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start = time.time()
print("Starting training...")
sys.stdout.flush()

n = len(train_loader) # number of batches

# Save initial output
save_model_output(model, test_loader, f"[{1},{num_epochs}][{0},{n}]", device=device)

losses = []
iters = 0

for epoch in range(num_epochs):
    running_loss = 0.0
    batch = 0

    for expected_outputs, _ in train_loader:
        inputs = downsample_tensor(expected_outputs)

        inputs = inputs.to(device)
        expected_outputs = expected_outputs.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, expected_outputs)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        batch += 1
        if batch % 10 == 0 or batch == 1:
            print(f"Finished [{batch},{n}] loss: {loss.item()}")
            sys.stdout.flush()
        if batch % 40 == 0 or (batch == 1 and epoch == 0):
            save_model_output(model, test_loader, f"[{epoch + 1},{num_epochs}][{batch},{n}]", device=device)
            sys.stdout.flush()
        
        if iters % 10 == 0: # Only append every 10th loss
            losses.append(loss.item())
        
        iters +=1

    # Print the average loss for the epoch
    average_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1},{num_epochs}] Loss: {average_loss:.4f}")
    sys.stdout.flush()

end = time.time()
print(f"Training finished. Took {round((end - start) / 60)} minutes")
sys.stdout.flush()

# Save the trained model to a file
torch.save(model.state_dict(), model_filename)
print(f"Model saved to {model_filename}")

# Save the final output
save_model_output(model, test_loader, f"[{num_epochs},{num_epochs}][{n},{n}]", device=device)
print("Finished training!")

plt.title("Model loss over iterations")
plt.ylabel('Loss')
plt.xlabel('Iterations')
x = [i * 10 for i in range(len(losses))] # Show actual number of iterations
plt.plot(x, losses)
plt.savefig('lossplot.png')
plt.close()

print("Saved loss plot")