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
        # print(f"Epoch {epoch + 1}/{n_epochs}")
        train_loss = 0.0
        correct = 0
        total = 0
        
        model.train()  # Set the model to training mode

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
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
        # print(f"Train loss: {train_loss / len(train_loader):.2f} - Train accuracy: {accuracy:.2f}%")
        train_accuracies.append(accuracy)
        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        valid_loss = 0.0
        correct = 0
        total = 0
        
        model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                with torch.cuda.amp.autocast():
                    y_hat = model(x)
                    loss = criterion(y_hat, y)

                valid_loss += loss.item()
                _, predicted = y_hat.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

            accuracy = 100 * correct / total
            valid_loss /= len(valid_loader)
            
            # print(f"Valid loss: {valid_loss:.2f} - Valid accuracy: {accuracy:.2f}%")
            valid_accuracies.append(accuracy)
            valid_losses.append(valid_loss)

    return train_accuracies, valid_accuracies, train_losses, valid_losses