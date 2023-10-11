import torch
import torch.nn as nn
import torch.optim as optim
from modules import ESPCN
from dataset import get_train_and_validation_loaders
import time

# Create the model and load training data
model = ESPCN(upscale_factor=4, channels=1)
train_loader, validation_loader = get_train_and_validation_loaders()

# Move the model onto the GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model.to(device)

# Set training parameters
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

print("Started training...")
for epoch in range(num_epochs):
    start_time = time.time()

    # Training Loop
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}", end=" - ")
    
    # Validation Loop
    model.eval()  # set the model to evaluation mode
    with torch.no_grad():
        val_loss = 0.0
        for i, data in enumerate(validation_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
    val_loss = val_loss / len(validation_loader)
    print(f"Validation Loss: {val_loss:.4f}", end=" - ")

    end_time = time.time()  # End time of epoch
    epoch_duration = end_time - start_time
    print(f"Completed in {epoch_duration:.2f} seconds.")

print("Finished training.")

# Save the final model
torch.save(model.state_dict(), 'final_model.pth')