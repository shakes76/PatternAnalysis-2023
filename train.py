import matplotlib.pyplot as plt
import torch
import numpy as np
import dataset
import module

import logging

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load training data with train valid split of 0.8 to 0.2
    train_data, valid_data, test_data = dataset.get_loaders()

    sample, _ = next(iter(train_data))

    # Print the shape of the input image
    print("Input image shape:", sample.shape)

    logging.info("Loaded data")

    # Create model and training components
    model, optimizer, criterion, scheduler = module.create_model(
        input_shape=(256, 256),            # Reduce input image shape
        latent_dim=8,                   # Smaller latent space dimension
        embed_dim=16,                     # Smaller image patch dimension
        attention_mlp_dim=16,            # Smaller dimension for cross-attention's feedforward network
        transformer_mlp_dim=16,          # Smaller dimension for the latent transformer's feedforward network
        transformer_heads=1,              # Fewer attention heads for the latent transformer
        dropout=0.1,                      # Reduce dropout for lower memory usage
        transformer_layers=1,            # Fewer layers in the latent transformer
        n_blocks=1,                       # Fewer Perceiver blocks
        n_classes=2,                      # Number of target classes (binary classification)
        batch_size=16,                     # Further reduce batch size to save memory
        lr=0.0001,                        # Smaller learning rate for stability
    )

    model = model.to(device)

    EPOCHS = 30

    train_size = len(train_data.dataset)
    valid_size = len(valid_data.dataset)

    print(f"Train size: {train_size}\nValid size: {valid_size}")

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
            logging.info(f"Early stopping. No improvement in validation loss for {early_stopping_patience} epochs.")
            break  # Exit the training loop

    # Save the best model state
    torch.save(best_model_state, 'saved/best_model.pth')

    # Plot training history
    plot_training_history(history)

def train(model, data_loader, criterion, optimizer, device):
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

def plot_training_history(history):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(history['train_acc'])
    plt.plot(history['valid_acc'])
    plt.xlim([0, len(history['train_acc'])])
    plt.xticks(range(len(history['train_acc'])))
    plt.ylim([0, 1])
    plt.title('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.show()

    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(history['train_loss'])
    plt.plot(history['valid_loss'])
    plt.xlim([0, len(history['train_loss'])])
    plt.xticks(range(len(history['train_loss'])))
    plt.ylim([0, 1])
    plt.title('Loss')
    plt.legend(['Training', 'Validation'])
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
