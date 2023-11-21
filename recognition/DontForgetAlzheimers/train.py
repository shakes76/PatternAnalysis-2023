"""
Model Trainer and Tester.
"""

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from dataset import AlzheimerDataset
from dataset import transform
from modules import ViT

import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    model = ViT().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_dataset = AlzheimerDataset("AD_NC/train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Training for 20 epochs
    num_epochs = 20
    all_loss = []
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        model.train()  
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)


            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch Loss: {total_loss / len(train_loader):.4f}")
        all_loss.append(total_loss / len(train_loader))

    epochs = list(range(1, 21))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, all_loss, color='blue')
    plt.title("Cross Enthropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
    return model

    
if __name__ == '__main__':
    
    test_dataset = AlzheimerDataset("AD_NC/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)  

    model = train()
    torch.save(model.state_dict(), 'model')
    model.eval() 
    correct = 0
    total = 0

    # Testing
    with torch.no_grad():  
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy on test dataset: {100 * correct / total:.2f}%")

