import torch
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from modules import ViT
from dataset import get_dataloaders
from datetime import datetime
import math

def train(batch_size: int = 8, workers: int = 4, image_resize: int = 224, dataroot: str = "AD_NC", 
            num_epochs: int = 10, device: str = 'cuda'):
    train_loader, test_dataloader, validation_loader = get_dataloaders(batch_size=batch_size, 
                                                                       workers=workers, 
                                                                       image_resize=image_resize,
                                                                       dataroot=dataroot,
                                                                       rgb=False)
    visual_transformer = ViT()
    visual_transformer.to(device)
    visual_transformer.train()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # ----------------------------------------
    # Loss Function and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(params=visual_transformer.parameters(), lr=1e-5, weight_decay=0.0001)

    # ----------------------------------------
    # Training loop
    train_loss_values = []
    val_acc_values = []
    start_time = time.time()
    print("Starting training loop") 
    
    batch_loss = 0.
    
    last_loss = 0.
    best_vloss = float('inf')
    for epoch in range(num_epochs):
        running_loss = 0.
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimiser.zero_grad()
            outputs = visual_transformer(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            batch_loss += loss.item()
            if (index) % batch_size == batch_size - 1:
                running_time = time.time()
                last_loss = batch_loss / float(batch_size)
                print("Epoch [{}/{}], Batch {} Loss: {:.5f}".format(epoch+1, num_epochs, index+1, last_loss))
                print(f"Timer: {running_time - start_time}")
                batch_loss = 0.

        average_loss = running_loss / len(train_loader)
        train_loss_values.append(average_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}")

        # -----------------
        # Validation Loop
        visual_transformer.eval()
        val_acc = 0
        running_vloss = 0.0  # Validation loss logic taken from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        with torch.no_grad():  # Disable gradient computation
            correct = 0
            total = 0
            for index, (inputs, labels) in enumerate(validation_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                validation_outputs = visual_transformer(inputs)
                validation_loss = criterion(validation_outputs, labels)
                running_vloss += validation_loss
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                _, predicted = torch.max(validation_outputs.data, 1)
        
        average_vloss = running_vloss / (index + 1)
        print(f"Validation Accuracy: {total / correct}, Validation loss: {average_vloss}")
        val_acc_values.append(total / correct)

        if average_vloss < best_vloss:
            best_vloss = average_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(visual_transformer.state_dict(), model_path)

    print(f"Finished Training")
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_values)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(val_acc_values)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig('training_plot.png', format='png')
    plt.show()
    torch.save(visual_transformer.state_dict(), "visual_transformer")


    # ----------------------------------------
    # Testing loop
    print("Testing...")
    start = time.time()
    visual_transformer.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = visual_transformer(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Test Accuracy: {} %'.format(100 * correct / total))
    end = time.time()
    print(f"Testing took: {end - start}")

def main():
    train()

if __name__ == '__main__':
    main()