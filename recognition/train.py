import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ISICDataset, get_transform
from modules import UNet2D
import matplotlib.pyplot as plt

# Parameters
batch_size = 2
initial_lr = 5e-4
l2_weight_decay = 1e-5
num_epochs = 30  # Adjusted for illustration

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Datasets & Loaders
train_dataset = ISICDataset(dataset_type='training', transform=get_transform())
val_dataset = ISICDataset(dataset_type='validation', transform=get_transform())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Criterion, Optimizer
model = UNet2D(in_channels=3, num_classes=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=l2_weight_decay)

# Custom Dice Loss
def dice_loss(output, target):
    smooth = 1e-6
    intersection = (output * target).sum(dim=(1, 2, 3))
    denominator = (output + target).sum(dim=(1, 2, 3))
    loss = 1 - (2.0 * intersection + smooth) / (denominator + smooth)
    return loss.mean()

def binary_dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice


# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    # Adjust learning rate
    lr = initial_lr * (0.985 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data['image'].to(device), data['mask'].to(device)
        optimizer.zero_grad()
        # outputs = model(inputs)
        outputs = torch.sigmoid(model(inputs))  # Applying sigmoid activation
        loss = binary_dice_loss(torch.softmax(outputs, dim=1), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 10 == 9:
            print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
            running_loss = 0.0

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data['image'].to(device), data['mask'].to(device)
            # outputs = model(inputs)
            outputs = torch.sigmoid(model(inputs))  # Applying sigmoid activation
            loss = binary_dice_loss(outputs, labels)
            # loss = dice_loss(torch.softmax(outputs, dim=1), labels)
            val_loss += loss.item()

    print(f"Validation Loss after epoch {epoch + 1}: {val_loss / len(val_loader):.3f}")

    # Save model checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"unet2d_model_epoch_{epoch + 1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f"Model checkpoint saved at {checkpoint_path}")

    # Visualization after each epoch (or every N epochs)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        # Retrieve a batch of data from the validation loader
        dataiter = iter(val_loader)
        data = next(dataiter)
        inputs, labels = data['image'].to(device), data['mask'].to(device)
        
        outputs = model(inputs)  # Model prediction
        
        # Optional: De-normalize image, if normalization is used in pre-processing.
        # inputs = denormalize(inputs)
        
        # Plotting the original image
        plt.subplot(1, 3, 1)
        plt.imshow(inputs[0].permute(1, 2, 0).cpu().numpy())
        plt.title("Input")
        plt.axis('off')
        
        # Plotting the actual mask
        plt.subplot(1, 3, 2)
        plt.imshow(labels[0].cpu().squeeze().numpy(), cmap='gray')
        plt.title("Actual")
        plt.axis('off')

        # Plotting the predicted mask
        plt.subplot(1, 3, 3)
        plt.imshow(outputs[0].cpu().squeeze().numpy(), cmap='gray')
        plt.title("Prediction")
        plt.axis('off')

        plt.show()
        
    model.train()  # Return model to training mode


# Save the final model
torch.save(model.state_dict(), "unet2d_model_final.pth")
