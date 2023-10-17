import torch
import torch.optim as optim
import torch.nn as nn
import math
from dataset import load_data
from modules import Network
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 100
learning_rate = 0.001

model = Network(upscale_factor=4, channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()


def train(epochs: int):
    model.train()
    running_loss = 0.0

    for epoch in range(1, epochs + 1):
        for input, label in dataloader:

            input.to(device)

            optimizer.zero_grad()

            outputs = model(input)
            loss = criterion(outputs, input)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {running_loss/len(dataloader)}")

        # Save model
        if epoch in [20, 40, 60, 80, 100]:
            print("Saving model on epoch " + str(epoch) + "...")
            torch.save(model.state_dict(), f'./Trained_Model.pth')
            print("Model saved!")

if __name__ == '__main__':
    dataloader = load_data()

    start_time = time.time()
    print("Starting training...")
    train(num_epochs)
    end_time = time.time()
    print("Training finished!")
    print(f"Time taken: {end_time - start_time} seconds, or {(end_time - start_time) / 60} minutes.")
