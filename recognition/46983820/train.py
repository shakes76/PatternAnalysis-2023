import matplotlib.pyplot as plt
import torch
import numpy as np
import dataset 
import module
import logging

"""    
Training the Perceiver Transformer for Alzheimer's classification using PyTorch.
"""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data with train valid split of 0.8 to 0.2
    train_data, valid_data, test_data = dataset.get_loaders()
    print(f"Train Data Size: {len(train_data.dataset)}\nValid Data Size: {len(valid_data.dataset)}")
    print(f"1st Train Data: {next(iter(train_data))[0].shape}\n1st Valid Data: {next(iter(valid_data))[0].shape}")

    sample, _ = next(iter(train_data))

    # Print the shape of the input image
    print("Input image shape:", sample.shape)

    logging.info("Loaded data")

    # Create model and training components
    model, optimizer, criterion, scheduler = module.create_model(
        input_shape=(256, 256),
        latent_dim=8, # Increase latent space dimension for more representational capacity
        embed_dim=16,
        attention_mlp_dim=16,
        transformer_mlp_dim=16,
        transformer_heads=4, # Use more attention heads for enhanced feature capturing
        dropout=0.1,
        transformer_layers=4,
        n_blocks=4,
        n_classes=2,
        lr=0.005,
    )

    model = model.to(device)

    EPOCHS = 30

    # Tracking minimum loss
    min_valid_loss = np.inf

    # Tracking accuracy and loss during training
    history = {'train_loss': [], 'train_acc': [], 'valid_loss': [], 'valid_acc': []}

    early_stopping_patience = 5
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        # Free up GPU memory before each epoch
        torch.cuda.empty_cache()

        train_loss, train_acc = train(model, train_data, criterion, optimizer, device)
        valid_loss, valid_acc = validate(model, valid_data, criterion, device)

        # Append metric history
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['valid_acc'].append(valid_acc)
        history['valid_loss'].append(valid_loss)

        logging.info(f"Epoch: {epoch + 1}\nTrain loss: {train_loss}\nTrain Accuracy: {train_acc}\nValid Loss: {valid_loss}\nValid Accuracy: {valid_acc}\n")

        scheduler.step()  # Step the learning rate scheduler after each epoch

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_model_state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': valid_loss
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            logging.info(f"No improvement in validation loss for {early_stopping_patience} epochs.")
            break  # Exit the training loop

    # Save the best model state
    torch.save(best_model_state, 'saved/best_model.pth')

def train(model, data_loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train.
        data_loader (torch.utils.data.DataLoader): The data loader for the training set.
        criterion (torch.nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimiser.
        device (torch.device): The device to use for training.
    
    Returns:
        loss (float): The training loss.
        accuracy (float): The training accuracy.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = len(data_loader.dataset)

    for batch, labels in data_loader:
        batch, labels = batch.to(device), labels.to(device)
        optimizer.zero_grad()
        prediction = model(batch)
        loss = criterion(prediction, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch)
        correct_predictions += torch.sum(torch.argmax(prediction, dim=1) == labels)

    loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return loss, accuracy

def validate(model, data_loader, criterion, device):
    """
    Validate the model on the validation set for one epoch.
    
    Args:
        Args:
        model (torch.nn.Module): The model to validate.
        data_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        criterion (torch.nn.Module): The loss function.
        device (torch.device): The device to use.
    
    Returns:
        loss (float): The validation loss.
        accuracy (float): The validation accuracy.
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = len(data_loader.dataset)

    with torch.no_grad():
        for batch, labels in data_loader:
            batch, labels = batch.to(device), labels.to(device)
            prediction = model(batch)
            loss = criterion(prediction, labels)

            total_loss += loss.item() * len(batch)
            correct_predictions += torch.sum(torch.argmax(prediction, dim=1) == labels)

    loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return loss, accuracy

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    main()
