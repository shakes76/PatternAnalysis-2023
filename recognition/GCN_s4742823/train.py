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

def train_model(model, data):    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    best_model_state = None
    best_accuracy = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # ----- Training -----
    print("--- Training ---")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        out = model(data)  # Pass the whole graph in.
        loss = criterion(out[data.train_mask].to(device), data.y[data.train_mask].to(device))  # Only calculate loss with train nodes.
        _, predicted = torch.max(out[data.train_mask], 1)
        predicted = predicted.cpu().numpy()
        y_true = data.y[data.train_mask].cpu().numpy()
        accuracy = accuracy = accuracy_score(y_true, predicted)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
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
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = copy.deepcopy(model.state_dict())
        val_losses.append(loss.item())
        val_accuracies.append(accuracy.item())

        if (epoch % 25 == 0):
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    if (best_model_state is not None):
        model.load_state_dict(best_model_state)

    # Save the model
    torch.save(model, "Facebook_GCN.pth") 
    return model

def test_model(model, data):
    # ----- Testing -----
    print("--- Testing ---")
    model.eval()
    with torch.no_grad():
        out = model(data)
        out_np = out.cpu().numpy()  # To be used for t-SNE
        _, predicted = torch.max(out[data.test_mask], 1)  # Get classes with the highest probablities (note that we only use test nodes).
        predicted = predicted.cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()
        accuracy = accuracy_score(y_true, predicted)
    print(f"Test Accuracy: {100 * accuracy:.2f}%")
    plot_tsne(out_np, y_true, data)

def plot_tsne(output, y_true, data):
    # Plotting t-SNE
    tsne = TSNE(n_components=2, perplexity=50)  # It was found that divergence did not converge before 50 perplexity
    transformed = tsne.fit_transform(output)

    plt.figure(figsize=(10, 8))
    for class_idx in range(NUM_CLASSES):
        mask = data.test_mask.cpu().numpy()
        plt.scatter(transformed[mask, 0][y_true == class_idx], transformed[mask, 1][y_true == class_idx], label=CLASSES[class_idx])
    plt.legend()
    plt.title("t-SNE Plot")
    plt.savefig("tsne_plot.png")

def plot_loss(train_losses, val_losses):
    # Plotting Loss
    plt.figure(figsize=(10, 8))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Losses')
    plt.savefig("loss.png")

def plot_accuracy(train_accuracies, val_accuracies):
    # Plotting Accuracy
    plt.figure(figsize=(10, 8))
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(val_accuracies, label='Validation Accuracy', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracies')
    plt.savefig("accuracy.png")

if __name__ == "__main__":
    # test / validation split
    test_size = 0.1
    val_size = 0.1

    # load_data() can take a filepath, otherwise will use default filepath in method.
    data = load_data(test_size=test_size, val_size=val_size)
    data = data.to(device)

    # Hyperparameters
    num_epochs = 500
    num_features = data.features.shape[1]  # 128 for default data
    hidden_dim = 64
    learning_rate = 1e-2
    dropout_prob = 0.5

    model = Model(num_features, hidden_dim, NUM_CLASSES, dropout_prob)
    model = model.to(device)

    trained_model = train_model(model, data)
    test_model(trained_model, data)