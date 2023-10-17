from dataset import load_data
from modules import Model
import torch
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", str(device))

test_size = 0.1
val_size = 0.1

# load_data() can take a filepath, otherwise will use default filepath in method.
data = load_data(test_size=test_size, val_size=val_size)
data = data.to(device)

num_epochs = 500
num_features = data.features.shape[1]  # 128 for default data
hidden_dim = 64
classes = ["Politicians", "Governmental Organisations", "Television Shows", "Companies"]
num_classes = len(classes)
learning_rate = 1e-2
dropout_prob = 0.5

model = Model(num_features, hidden_dim, num_classes, dropout_prob)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)

best_model = None
best_accuracy = 0

# ----- Training -----
print("--- Training ---")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    out = model(data)  # Pass the whole graph in.
    loss = criterion(out[data.train_mask].to(device), data.y[data.train_mask].to(device))  # Only calculate loss with train nodes.
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    optimizer.zero_grad()

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
        best_model = model.state_dict()


    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# ----- Testing -----
print("--- Testing ---")
if (best_model is not None):
    model.load_state_dict(best_model)
model.eval()
with torch.no_grad():
    out = model(data)
    out_np = out.cpu().numpy()  # To be used for t-SNE
    _, predicted = torch.max(out[data.test_mask], 1)  # Get classes with the highest probablities (note that we only use test nodes).
    predicted = predicted.cpu().numpy()
    y_true = data.y[data.test_mask].cpu().numpy()
    accuracy = accuracy_score(y_true, predicted)
print(f"Test Accuracy: {100 * accuracy:.2f}%")

# Plotting t-SNE
tsne = TSNE(n_components=2, perplexity=50)  # It was found that divergence did not converge before 50 perplexity
transformed = tsne.fit_transform(out_np)

plt.figure(figsize=(10, 8))
for class_idx in range(num_classes):
    mask = data.test_mask.cpu().numpy()
    plt.scatter(transformed[mask, 0][y_true == class_idx], transformed[mask, 1][y_true == class_idx], label=classes[class_idx])
plt.legend()
plt.title("t-SNE Plot")
plt.savefig("tsne_plot.png")