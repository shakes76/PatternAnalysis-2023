""" Source code for training, validating, testing and saving the StyleGAN model. """

import torch
from tqdm import tqdm, trange

def train(model, train_loader, valid_loader, criterion, optimizer, device, n_epochs=10):
    """ Train the model. Plot the losses and accuracies during training. """
    # Construct scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create lists to store the losses and accuracies
    train_accuracies = []
    valid_accuracies = []
    train_losses = []
    valid_losses = []

    # Training loop
    for epoch in trange(n_epochs, desc="Training"):
        train_loss = 0.0
        correct = 0
        total = 0

        # Set the model to training mode
        model.train()
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

        train_accuracies.append(accuracy)
        train_losses.append(train_loss / len(train_loader))

        # Validation loop
        valid_loss = 0.0
        correct = 0
        total = 0

        # Set the model to evaluation mode
        model.eval()  
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

            valid_accuracies.append(accuracy)
            valid_losses.append(valid_loss)

    return train_accuracies, valid_accuracies, train_losses, valid_losses

def test(model, device, test_loader, criterion):
    """ Test the model on the test set.

    Args:
        model (Module): The trained model
        device (str): Device to run the inference on
        test_loader (Dataloader): The dataloader containing the test data
        criterion (Criterion): Loss function
    """

    # Set the model to evaluation mode
    model.eval()
    
    # Test loop
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            with torch.cuda.amp.autocast():
                y_hat = model(x)
                loss = criterion(y_hat, y)

            test_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        test_loss /= len(test_loader)  # Calculate average test loss
        accuracy = 100 * correct / total  # Calculate accuracy
        
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {accuracy:.2f}%")