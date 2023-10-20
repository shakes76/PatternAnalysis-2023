import torch
import torch.optim as optim
import torch.nn as nn
from dataset import load_test_data, load_train_data
from modules import Network
import time
import matplotlib.pyplot as plt
from utils import *
from torch.optim.lr_scheduler import StepLR

model = Network(upscale_factor=upscale_factor, channels=channels, dropout_probability=dropout_probability).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
criterion = nn.MSELoss()

def train(epochs: int):
    epoch_train_losses = []
    epoch_val_losses = []

    epoch_train_psnrs = []
    epoch_val_psnrs = []

    train_loader = load_train_data(train_path)
    val_loader = load_test_data(test_path)

    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0

        for target, _ in train_loader:
            target = target.to(device)
            input = down_sample(target) 

            optimizer.zero_grad()

            outputs = model(input)
            loss = criterion(outputs, target) 

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        epoch_train_losses.append(avg_train_loss)

        # Validation loss
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for target, _ in val_loader:
                target = target.to(device)
                input = down_sample(target)
                outputs = model(input)
                loss = criterion(outputs, target)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        epoch_val_losses.append(avg_val_loss)

        avg_train_psnr = compute_psnr(avg_train_loss)
        epoch_train_psnrs.append(avg_train_psnr.item())
    
        avg_val_psnr = compute_psnr(avg_val_loss)
        epoch_val_psnrs.append(avg_val_psnr.item())

        print(f"Epoch [{epoch}/{epochs}] Train Loss: {avg_train_loss}, Train PSNR: {avg_train_psnr}, Validation Loss: {avg_val_loss}, Validation PSNR: {avg_val_psnr}")

        # Update the learning rate
        scheduler.step()

        # Save model
        if epoch in [20, 40, 60, 80, 100]:
            print(f"Saving model on epoch {epoch}...")
            torch.save(model.state_dict(), f'./models/Trained_Model_Epoch_{epoch}.pth')
            print("Model saved!")

    return epoch_train_losses, epoch_val_losses, epoch_train_psnrs, epoch_val_psnrs

if __name__ == '__main__':
    start_time = time.time()
    print("Starting training...")
    train_losses, val_losses, epoch_train_psnrs, epoch_val_psnrs = train(num_epochs)
    end_time = time.time()
    print("Training finished!")
    print(f"Time taken: {end_time - start_time} seconds, or {(end_time - start_time) / 60} minutes.")

    # Plotting the losses
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Time')
    plt.legend()
    plt.savefig('figures/losses.png')
    plt.figure()
    plt.plot(range(1, num_epochs + 1), epoch_train_psnrs, label='Training PSNR')
    plt.plot(range(1, num_epochs + 1), epoch_val_psnrs, label='Validation PSNR')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.title('Training and Validation PSNR over Time')
    plt.legend()
    plt.savefig('figures/psnrs.png')

