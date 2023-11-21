'''
This file contains the hyperparameters, as well as the functions used to train and test the model, and to plot its results.
It also has the main functionality, which is to create a new Model, train it, test it, and plot the results.
This file will save the model as "Facebook_GCN.pth", which can then be loaded and tested in predict.py.
'''
from dataset import load_data
from modules import Model
import torch
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils import SEED, device, NUM_CLASSES, CLASSES
import copy

# Set seed for reproducibility.
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Hyperparameters
NUM_EPOCHS = 500
HIDDEN_DIM = 64
LEARNING_RATE = 1e-2
DROPOUT_PROB = 0.5
# test / validation split (10% each default)
TEST_SIZE = 0.1
VAL_SIZE = 0.1

def train_model(model, data):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    best_model_state = None
    best_accuracy = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # ----- Training -----
    print("--- Training ---")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        epoch_acc = 0

        out = model(data)  # Pass the whole graph in.
        loss = criterion(out[data.train_mask].to(device), data.y[data.train_mask].to(device))  # Only calculate loss with train nodes.
        _, predicted = torch.max(out[data.train_mask], 1)
        predicted = predicted.cpu().numpy()
        y_true = data.y[data.train_mask].cpu().numpy()
        accuracy = accuracy = accuracy_score(y_true, predicted)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += accuracy.item()
        optimizer.zero_grad()
        train_losses.append(loss.item())
        train_accuracies.append(accuracy.item())

        # --- Validation ---
        model.eval()

        out = model(data)
        loss = criterion(out[data.val_mask], data.y[data.val_mask])
        _, predicted = torch.max(out[data.val_mask], 1)
        predicted = predicted.cpu().numpy()
        y_true = data.y[data.val_mask].cpu().numpy()
        accuracy = accuracy_score(y_true, predicted)
        # Store the most accurate model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = copy.deepcopy(model.state_dict())
        val_losses.append(loss.item())
        val_accuracies.append(accuracy.item())

        if (epoch % 25 == 0):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    if (best_model_state is not None):
        model.load_state_dict(best_model_state)

    # Plot results
    plot_accuracy(train_accuracies, val_accuracies)
    plot_loss(train_losses, val_losses)

    # Save the model
    torch.save(model, "Facebook_GCN.pth") 
    return model

def test_model(model, data):
    # ----- Testing -----
    print("--- Testing ---")
    model.eval()
    with torch.no_grad():
        out = model(data)
        out_np = out.cpu().numpy()  # To be used for t-SNE.
        _, predicted = torch.max(out[data.test_mask], 1)  # Get classes with the highest probablities (note that we only use test nodes).
        predicted = predicted.cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()
        accuracy = accuracy_score(y_true, predicted)
    print(f"Test Accuracy: {100 * accuracy:.2f}%")
    plot_tsne(out_np, y_true, data)

def plot_tsne(output, y_true, data):
    # Plotting t-SNE
    tsne = TSNE(n_components=2, perplexity=30)
    transformed = tsne.fit_transform(output)

    plt.figure(figsize=(10, 8))
    for class_idx in range(NUM_CLASSES):
        mask = data.test_mask.cpu().numpy()  # Use test data mask stored in GCNData object.
        plt.scatter(transformed[mask, 0][y_true == class_idx], 
                    transformed[mask, 1][y_true == class_idx], 
                    label=CLASSES[class_idx])  # Plot all transformed nodes for each class.
    plt.legend()
    plt.title("t-SNE Plot")
    plt.savefig("tsne_plot.png")

def plot_loss(train_losses, val_losses):
    # Plotting Loss
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig("loss.png")

def plot_accuracy(train_accuracies, val_accuracies):
    # Plotting Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracies')
    plt.savefig("accuracy.png")

if __name__ == "__main__":
    # load_data() can take a filepath, otherwise will use default filepath in method.
    data = load_data(test_size=TEST_SIZE, val_size=VAL_SIZE)
    data = data.to(device)

    num_features = data.features.shape[1]  # 128 for default data

    model = Model(num_features, HIDDEN_DIM, NUM_CLASSES, DROPOUT_PROB)
    model = model.to(device)

    trained_model = train_model(model, data)
    test_model(trained_model, data)