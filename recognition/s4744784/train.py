import torch
import torch.optim as optim
import torch.nn as nn
import math
from dataset import load_data
from modules import Network
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Network(upscale_factor=4, channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()


def train(epochs=100):
    model.train()
    for epoch in range(epochs):
        for (inputs_low_res, _), (targets_high_res, _) in dataloader:
            inputs_low_res = inputs_low_res.to(device)
            targets_high_res = targets_high_res.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs_low_res)
            loss = criterion(outputs, targets_high_res)
            loss.backward()
            optimizer.step()
        
        # Print PSNR (Peak Signal-to-Noise Ratio)
        psnr = 10 * math.log10(1 / loss.item())
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item()} PSNR: {psnr}")
        
        # Save model
        
        if epoch % 20 == 0:
            print("Saving model...")
            torch.save(model.state_dict(), f'./espcn_epoch_{epoch}.pth')
            print("Model saved!")

if __name__ == '__main__':
    dataloader = load_data()

    start_time = time.time()
    print("Starting training...")
    train()
    end_time = time.time()
    print("Training finished!")
    print(f"Time taken: {end_time - start_time} seconds, or {(end_time - start_time) / 60} minutes.")
