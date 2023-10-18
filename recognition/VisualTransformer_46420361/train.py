"""Used to train and evaluate the model performance"""
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from dataset import load_dataloaders
from modules import ViT

# device config
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found, using CPU")
    return device

def create_model(image_size, in_channels, patch_size, embedding_dims, num_heads, num_classes, patches):
    return ViT(img_size=image_size, 
               in_channels=in_channels,
               patch_size=patch_size,
               embedding_dims=embedding_dims,
               num_heads=num_heads,
               num_classes=num_classes,
               patches=patches)

def train_model(model, root, image_size, batch_size, crop_size, learning_rate, weight_decay, epochs):
    device = get_device()
    train_loader, _ = load_dataloaders(root, image_size, crop_size, batch_size)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    model.train()
    
    for epoch in trange(epochs, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")

def evaluate_model(model, root, image_size, crop_size, batch_size):
    device = get_device()
    _, test_loader = load_dataloaders(root, image_size, crop_size, batch_size)
    criterion = CrossEntropyLoss()
    model.eval()
    
    with torch.no_grad():
        correct, total = 0, 0
        validation_loss = 0.0
        for batch in tqdm(test_loader, desc="Validation"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            validation_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Validation loss: {validation_loss:.2f}")
        print(f"Validation accuracy: {correct / total * 100:.2f}%")
    return

def predict(model, dataloader, num_samples=5):
    # Set the model to evaluation mode
    model.eval()

    # randomly sample images 
    samples = []
    for _ in range(num_samples):
        data, _ = next(iter(dataloader))  # You may need to adapt this based on your dataloader structure
        samples.append(data)

    # Make predictions
    with torch.no_grad():
        predictions = model(torch.stack(samples))  # Assuming your model takes a batch as input

    # Plot the predictions
    for i in range(num_samples):
        plt.figure()
        plt.imshow(samples[i][0].permute(1, 2, 0).cpu().numpy())  # Assuming your input is an image tensor
        plt.title(f"Prediction: {predictions[i].argmax()}")
        plt.show()

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
        