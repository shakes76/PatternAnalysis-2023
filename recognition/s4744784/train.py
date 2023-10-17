import torch
import torch.optim as optim
import torch.nn as nn
import math
from dataset import load_data
from modules import Network
import time
import matplotlib.pyplot as plt
from utils import *

model = Network(upscale_factor=upscale_factor, channels=channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

def train(epochs: int):
    model.train()
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        running_loss = 0.0

        for input, _ in dataloader:
            input = input.to(device)

            optimizer.zero_grad()

            outputs = model(input)
            loss = criterion(outputs, input)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss)
        
        print(f"Epoch [{epoch}/{epochs}] Loss: {avg_epoch_loss}")

        # Save model
        if epoch in [20, 40, 60, 80, 100]:
            print(f"Saving model on epoch {epoch}...")
            torch.save(model.state_dict(), f'./Trained_Model_Epoch_{epoch}.pth')
            print("Model saved!")

    # Plotting the losses
    plt.figure()
    plt.plot(range(1, epochs + 1), epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')
    plt.legend()
    plt.savefig('training_loss.png')

if __name__ == '__main__':
    dataloader = load_data()

    start_time = time.time()
    print("Starting training...")
    train(num_epochs)
    end_time = time.time()
    print("Training finished!")
    print(f"Time taken: {end_time - start_time} seconds, or {(end_time - start_time) / 60} minutes.")
