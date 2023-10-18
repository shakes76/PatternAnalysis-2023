import torch
import torch.nn as nn
import torch.optim as optim
from modules import ImprovedUNet
from dataset import UNetData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to images
path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ISIC2018/'
    
# Hyper-parameters
num_epochs = 1
learning_rate = 0.1
image_height = 512 
image_width = 512

def main():
    # Improved UNet model
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)   

    # Training data
    data = UNetData(path=path, height=image_height, width=image_width)
    train_data = data.train_loader

    # Binary class. loss function and Adam optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        for (image, targets) in train_data:
            image = image.to(device=device)
            targets = targets.float().to(device=device) 

            # automatic mixed-precision training
            with torch.cuda.amp.autocast():
                predictions = model(image)
                loss = loss_fn(predictions, targets) 

            # backpropagation 
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")



if __name__ == "__main__":
    main()