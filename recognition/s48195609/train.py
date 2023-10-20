"""
train.py

Code for training, validating, testing, and saving the model.

Author: Your Name
Date Created: Your Date
"""
import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from datasplit import load_data
from modules import build_vision_transformer
from parameter import *
from torchsummary import summary
model = build_vision_transformer(INPUT_SHAPE, IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, NUM_HEADS, PROJECTION_DIM, HIDDEN_UNITS, DROPOUT_RATE, NUM_LAYERS, MLP_HEAD_UNITS, LOCAL_WINDOW_SIZE)
model.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
summary(model, INPUT_SHAPE)

def compile_model():
    """
    Builds and compiles the model.
    """
    # Build and compile model
    model = build_vision_transformer(INPUT_SHAPE, IMAGE_SIZE, PATCH_SIZE, NUM_PATCHES, NUM_HEADS, PROJECTION_DIM, HIDDEN_UNITS, DROPOUT_RATE, NUM_LAYERS, MLP_HEAD_UNITS, LOCAL_WINDOW_SIZE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    return model, optimizer, criterion

def train_model(model, optimizer, criterion, train_data, val_data):
    """
    Trains and saves the model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    # Define data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=BATCH_SIZE)

    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        running_train_loss = 0.0
        running_val_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.view(-1, 1).float())

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_labels.view(-1, 1).float())

                running_val_loss += val_loss.item()

        model.train()

        train_loss = running_train_loss / len(train_loader)
        val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Save model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # Plot and save loss curves
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('losses.png')

if __name__ == '__main__':
    # Load data
    train, val, test = load_data()

    # Compile and train model
    model, optimizer, criterion = compile_model()
    train_model(model, optimizer, criterion, train, val)
