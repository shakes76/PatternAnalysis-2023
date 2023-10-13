import torch
import torch.optim as optim
from modules import ESPCN
from dataset import get_dataloaders

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate_initial = 0.01
learning_rate_final = 0.0001
epochs = 20  # You can adjust this based on your needs
upscale_factor = 4  # Adjust based on your needs

# Load datasets
train_loader, test_loader = get_dataloaders("C:\\Users\\soonw\\ADNI\\AD_NC")  # Replace 'path_to_root_dir' with your dataset path

# Initialize model and move to device
model = ESPCN(upscale_factor).to(device)

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate_initial)

# Learning rate scheduler to adjust learning rate during training
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=learning_rate_final)

# Training loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (LR, HR, _) in enumerate(train_loader):
        LR, HR = LR.to(device), HR.to(device)

        # Forward pass
        outputs = model(LR)
        loss = criterion(outputs, HR)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Adjust learning rate
    scheduler.step(loss)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

print("Training finished.")