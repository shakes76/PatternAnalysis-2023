import torch
import torch.optim as optim
import torch.nn as nn
import time
from modules import ViT
from dataset import get_dataloaders

def train_and_test(num_epochs: int, workers: int, device, batch_size: int):
    visual_transformer = ViT(workers).to(device)
    train_loader, test_loader = get_dataloaders(batch_size, workers)
    torch.save(visual_transformer.state_dict(), "visual_transformer")
    
    # ----------------------------------------
    # Loss Function and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(params=visual_transformer.parameters(), lr=3e-3, weight_decay=0.3)

    # ----------------------------------------
    # Training loop
    visual_transformer.train()
    start_time = time.time()
    print("Starting training loop")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimiser.zero_grad()
            outputs = visual_transformer(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if (index+1) % 2 == 0:
                running_time = time.time()
                print("Epoch [{}/{}], Loss: {:.5f}".format(epoch+1, num_epochs, loss.item()))
                print(f"Timer: {running_time - start_time}")
                running_loss = 0.0

    print(f"Finished Training")

    # ----------------------------------------
    # Testing loop
    print("Testing...")
    start = time.time()
    visual_transformer.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = visual_transformer(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Test Accuracy: {} %'.format(100 * correct / total))
    end = time.time()
    print(f"Testing took: {end - start}")
