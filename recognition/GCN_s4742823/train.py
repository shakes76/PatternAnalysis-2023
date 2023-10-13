from dataset import load_data
from modules import Model
import torch
from sklearn.metrics import accuracy_score

print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_size = 0.1
val_size = 0.1

# load_data() can take a filepath, otherwise will use default filepath in method.
data, features = load_data(test_size=test_size, val_size=val_size) # Get features to get their shape.
data = data.to(device)

num_epochs = 500
num_features = features.shape[1]  # 128 for default data
hidden_dim = 64
num_classes = 4  # politicians, governmental organizations, television shows and companies
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
    loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Only calculate loss with train nodes.
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    optimizer.zero_grad()

    # --- Validation ---
    model.eval()

    out = model(data)
    loss = criterion(out[data.val_mask], data.y[data.val_mask])
    _, predicted = torch.max(out[data.val_mask], 1)
    accuracy = accuracy_score(data.y[data.val_mask], predicted)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model.state_dict()


    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

# ----- Testing -----
print("--- Testing ---")
model.eval()
with torch.no_grad():
    out = model(data)
    _, predicted = torch.max(out[data.test_mask], 1)  # Get classes with the highest probablities (note that we only use test nodes).
    accuracy = accuracy_score(data.y[data.test_mask], predicted)
print(f"Test Accuracy: {100 * accuracy:.2f}%")