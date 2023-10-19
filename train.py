import torch
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from modules import ViT
from dataset import get_dataloaders
from datetime import datetime

def train(batch_size: int, workers: int, image_resize: int, dataroot: str, 
            num_epochs: int, device: str = 'cuda'):
    train_loader, test_dataloader, validation_loader = get_dataloaders(batch_size=batch_size, 
                                                                       workers=workers, 
                                                                       image_resize=image_resize,
                                                                       dataroot=dataroot)
    visual_transformer = ViT()
    visual_transformer.to(device)
    visual_transformer.train()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # ----------------------------------------
    # Loss Function and Optimiser
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(params=visual_transformer.parameters(), lr=(1e-3)/(4096 // batch_size), weight_decay=0.03)

    # ----------------------------------------
    # Training loop
    train_loss_values = []
    val_acc_values = []
    start_time = time.time()
    print("Starting training loop") 
    
    running_loss = 0.
    last_loss = 0.
    best_vloss = float('inf')
    for epoch in range(num_epochs):
        for index, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimiser.zero_grad()
            outputs = visual_transformer(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimiser.step()

            running_loss += loss.item()
            if (index) % batch_size == batch_size - 1:
                running_time = time.time()
                last_loss = running_loss / batch_size
                print("Epoch [{}/{}], Batch {} Loss: {:.5f}".format(epoch+1, num_epochs, index+1, last_loss))
                print(f"Timer: {running_time - start_time}")
                running_loss = 0.

        train_loss_values.append(last_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {last_loss}")

        # -----------------
        # Validation Loop
        visual_transformer.eval()
        val_acc = 0
        running_vloss = 0.0  # Validation loss logic taken from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
        with torch.no_grad():  # Disable gradient computation
            for index, (inputs, labels) in enumerate(validation_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                validation_outputs = visual_transformer(inputs)
                validation_loss = criterion(validation_outputs, labels)
                running_vloss += validation_loss

                _, predicted = torch.max(validation_outputs.data, 1)
                val_acc += (predicted == labels).sum().item() / len(outputs)
        
        average_vloss = running_vloss / (index + 1)
        print(f"Validation Accuracy: {val_acc / len(validation_loader)}, Validation loss: {average_vloss}")
        val_acc_values.append(val_acc / len(validation_loader))

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