""" Source code for training, validating, testing and saving the StyleGAN model. """

import torch
from tqdm import tqdm, trange

def train(model, train_loader, valid_loader, criterion, optimizer, device, n_epochs=10):
    # Construct scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    train_accuracies = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []

    # Training loop
    for epoch in trange(n_epochs, desc="Training"):
        train_loss = 0.0
        correct = 0
        total = 0
        batch_num = 0
        
        model.train()  # Set the model to training mode

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            batch_num += 1
            x, y = batch
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                y_hat = model(x)
                loss = criterion(y_hat, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch + 1}/{n_epochs} - Loss: {train_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
        train_accuracies.append(accuracy)
        train_losses.append(train_loss / len(train_loader))

            # Test loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(valid_loader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(valid_loader)

                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)
            print(f"Test loss: {test_loss:.2f}")
            print(f"Test accuracy: {correct / total * 100:.2f}%")
            valid_accuracies.append(correct / total * 100)
            valid_losses.append(test_loss)

    return train_accuracies, valid_accuracies, train_losses, valid_losses