from sklearn.model_selection import KFold
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import KFold
import torch
import modules
import train 

model, node_labels, all_features_tensor, adjacency_normed_tensor = train.getModel()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyperparameters
learning_rate = 0.01
epochs = 300
hidden_features = 64
out_features = 4  
n_splits = 10

kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)  # random_state for reproducibility

all_labels_tensor = torch.LongTensor(node_labels[:, 1]).to(device)

all_accuracies = []

for train_indices, test_indices in kf.split(all_features_tensor):

    # Create masks for train and test based on the current split
    train_mask_tensor = torch.BoolTensor([i in train_indices for i in range(len(all_features_tensor))]).to(device)
    test_mask_tensor = torch.BoolTensor([i in test_indices for i in range(len(all_features_tensor))]).to(device)

    train_labels_tensor = torch.LongTensor(node_labels[train_indices, 1]).to(device)
    
    # Model
    model = modules.GCN(in_features=all_features_tensor.shape[1], hidden_features=hidden_features, out_features=out_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(all_features_tensor, adjacency_normed_tensor)
        loss = F.nll_loss(out[train_mask_tensor], train_labels_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Fold: {len(all_accuracies)+1}, Epoch {epoch}, Loss: {loss.item()}")

    # Testing
    test_labels_tensor = torch.LongTensor(node_labels[test_indices, 1]).to(device)
    model.eval()
    with torch.no_grad():
        test_out = model(all_features_tensor, adjacency_normed_tensor)
        pred = test_out[test_mask_tensor].argmax(dim=1)
        correct = (pred == test_labels_tensor).sum().item()
        acc = correct / test_labels_tensor.size(0)
        all_accuracies.append(acc)
        print(f"Fold: {len(all_accuracies)}, Test accuracy: {acc * 100:.2f}%")

# Print average accuracy over all folds
print(f"Average accuracy over {n_splits}-fold cross-validation: {100 * sum(all_accuracies) / len(all_accuracies):.2f}%")






# Visualisation TSNE after model is trained
model.eval()
with torch.no_grad():
    # Do a forward pass to compute embeddings
    _ = model(all_features_tensor, adjacency_normed_tensor)  
    embeddings = model.get_embeddings().cpu().numpy()

tsne = TSNE(n_components=2, random_state=99)
embeddings_2d = tsne.fit_transform(embeddings)

number_of_classes = 4

plt.figure(figsize=(10, 8))
for label in range(number_of_classes):
    indices = np.where(node_labels[:, 1] == label)
    plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=str(label), s=5)
plt.legend()
plt.title('t-SNE visualization of GCN embeddings')
plt.show()
