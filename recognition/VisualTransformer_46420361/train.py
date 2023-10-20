"""Used to train and evaluate the model performance"""
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from dataset import load_dataloaders
from modules import ViT
from torch.nn.utils import clip_grad_norm_

# device config
def get_device():
    """gets the device (GPU) used for training. Prints warning if using CPU.

    Returns:
        string: the device that will be used for training
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found, using CPU")
    return device

def create_model(image_size, channels, patch_size, embedding_dims, num_heads, device):
    """Creates the vision transformer model based on parameters

    Args:
        image_size (int): the size of the image to be resized to before training
        channels (int): the number of channels in the image
        patch_size (int): the size of the patches used for splitting the image
        embedding_dims (int): the number of embedding dimensions
        num_heads (int): the number of heads used for training
        device (string): the device that will be used for training

    Returns:
        <class 'modules.ViT'>: the vision transformer model
    """
    model = ViT(
    img_size=image_size,
    in_channels=channels,
    patch_size=patch_size,
    embedding_dims=embedding_dims,
    num_heads=num_heads
    ).to(device)
    return model

def train_model(model, root, learning_rate, weight_decay, epochs, device):
    """trains and evaluates the model

    Args:
        model (<class 'modules.ViT'>): the vision transformer model
        root (string): the root of the dataset folder
        learning_rate (int): the rate at which the model learns (steps through cost function)
        weight_decay (int): the rate of decay of weights
        epochs (int): the number of epochs used in training
        device (string): the device used to train the model

    Returns:
        list: accuracies and losses for both training and evaluation
    """
    train_dataloader, _, validation_dataloader = load_dataloaders(root)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CrossEntropyLoss()

    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    clip_value = 1.0

    for epoch in trange(epochs, desc="Training"):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_dataloader)

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        accuracy = correct / total
        training_accuracies.append(accuracy)

        training_losses.append(train_loss)

        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f} Accuracy: {accuracy* 100:.2f}%")
        
        model.eval() # evaluation mode
        
        # Validation loop
        with torch.no_grad():
            correct, total = 0, 0
            test_loss = 0.0
            for batch in tqdm(validation_dataloader, desc="Testing"):
                x, y = batch
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                test_loss += loss.detach().cpu().item() / len(validation_dataloader)

                correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
                total += len(x)

            accuracy = correct / total
            validation_accuracies.append(accuracy)

            validation_losses.append(train_loss)

            print(f"Validation loss: {test_loss:.2f}")
            print(f"Validation accuracy: {correct / total * 100:.2f}%")

    return training_accuracies, training_losses, validation_accuracies, validation_losses
    

def test(model, criterion, device, test_dataloader):
    """tests model on test dataset

    Args:
        model (<class 'modules.ViT'>): the vision transformer model
        criterion (nn.CrossEntropyLoss): the criterion be used in the model
        device (string): the device being used by the model for testing
        test_dataloader (torch.utils.data.DataLoader): the test dataloader
    """
    model.eval() # evaluation mode
        
    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_dataloader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_dataloader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

def load_model(model_name):
    """loads saved model

    Args:
        model_name (string): name of the model to be loaded (.pth)

    Returns:
        <class 'modules.ViT'>: the loaded model
    """
    return torch.load(model_name)

def save_model(model, model_name):
    """saves model to current working directory

    Args:
        model (<class 'modules.ViT'>): the vision transformer model to be saved
        file (string): file name of the model to be saved (needs .pth)
    """
    torch.save(model, model_name)
        