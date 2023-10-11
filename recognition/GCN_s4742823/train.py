from dataset import load_data
from modules import Model
import torch

print("PyTorch Version:", torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load_data() can take a filepath, otherwise will use default filepath in method.
data, features = load_data() # Get features to get their shape.
data = data.to(device)

num_epochs = 200
num_features = features.shape[1] # 128 for default data
hidden_dim = 64
num_classes = 4 # politicians, governmental organizations, television shows and companies
learning_rate = 0.01

model = Model(num_features, hidden_dim, num_classes)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

model = model.to(device)

model.train()
for epoch in range(num_epochs):
    total_loss = 0

    out = model(data) # Pass the whole graph in.
    loss = criterion(out[data.train_mask], data.y[data.train_mask]) # Only calculate loss with train nodes.
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
    optimizer.zero_grad()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")