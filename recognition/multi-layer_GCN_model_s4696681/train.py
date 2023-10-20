import modules as modules
import dataset as dataset
import torch
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adjacency_normed = dataset.normalise_adjacency_matrix(dataset.create_adjacency_matrix())
feature_vectors = dataset.create_feature_vectors()
node_labels = dataset.convert_labels()
all_features_tensor, train_labels_tensor, test_labels_tensor, train_tensor, test_tensor, test_mask = dataset.create_tensors()
adjacency_normed_tensor = torch.FloatTensor(adjacency_normed)
adjacency_normed_tensor = torch.FloatTensor(adjacency_normed).to(device)

learning_rate = 0.01
epochs = 200
hidden_features = 64
out_features = 4  


model = modules.GCN(in_features=all_features_tensor.shape[1], hidden_features=hidden_features, out_features=out_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(all_features_tensor, adjacency_normed_tensor)
        loss = F.nll_loss(out[train_tensor], train_labels_tensor)

        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
    test_labels_tensor = torch.LongTensor(node_labels[test_mask, 1]).to(device)
    return model, test_labels_tensor


