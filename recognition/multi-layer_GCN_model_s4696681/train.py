import modules as modules
import dataset as dataset
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, ParameterGrid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

adjacency_normed = dataset.normalise_adjacency_matrix(dataset.create_adjacency_matrix())
feature_vectors = dataset.create_feature_vectors()
node_labels = dataset.convert_labels()
all_features_tensor, train_labels_tensor, test_labels_tensor, train_tensor, test_tensor, test_mask = dataset.create_tensors()
adjacency_normed_tensor = torch.FloatTensor(adjacency_normed)
adjacency_normed_tensor = torch.FloatTensor(adjacency_normed).to(device)

def create_new_tensors(train_indices, test_indices):
    node_ids = feature_vectors[:, 0]


    train_mask = np.isin(node_ids, train_indices)
    test_mask = np.isin(node_ids, test_indices)
    
    train_labels_tensor = torch.LongTensor(node_labels[train_mask, 1]).to(device)
    test_labels_tensor = torch.LongTensor(node_labels[test_mask, 1]).to(device)
    
    train_tensor = torch.BoolTensor(train_mask).to(device)
    test_tensor = torch.BoolTensor(test_mask).to(device)
    
    train_mask_tensor = torch.BoolTensor(train_mask).to(device)
    test_mask_tensor = torch.BoolTensor(test_mask).to(device)
    return train_tensor, test_tensor, train_labels_tensor, test_labels_tensor, train_mask_tensor, test_mask_tensor



out_features = 4
# define the ranges of your hyperparameters for the grid search
param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'epochs': [100, 200, 300],
    'hidden_features': [32, 64, 128]
}

# Create a list of parameter combinations to search
parameter_combinations = list(ParameterGrid(param_grid))

def train_and_evaluate(model, train_tensor1, test_tensor1, train_labels_tensor1, test_labels_tensor1, train_mask_tensor1, test_mask_tensor1, parameters):
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    hidden_features = parameters['hidden_features']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(all_features_tensor, adjacency_normed_tensor)  # or use train_features
        loss = F.nll_loss(out[train_tensor1], train_labels_tensor1)  # or use train_labels

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_out = model(all_features_tensor, adjacency_normed_tensor)
        pred = test_out[test_mask_tensor1].argmax(dim=1)
        correct = (pred == test_labels_tensor1).sum().item()
        acc = correct / test_labels_tensor1.size(0)
        print(f"Test accuracy: {acc * 100:.2f}%")
        test_accuracy = acc*100

    return test_accuracy 

# Set up K-fold cross-validation
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True)

best_parameters = None
best_accuracy = -1

# Iterate over each combination of parameters
for parameters in parameter_combinations:
    fold_accuracies = []

    for train_indices, test_indices in kf.split(all_features_tensor):
        model = modules.GCN(in_features=all_features_tensor.shape[1], hidden_features=parameters['hidden_features'], out_features=out_features).to(device)
        train_tensor1, test_tensor1, train_labels_tensor1, test_labels_tensor1, train_mask_tensor1, test_mask_tensor1 = create_new_tensors(train_indices, test_indices)
        accuracy = train_and_evaluate(model, train_tensor1, test_tensor1, train_labels_tensor1, test_labels_tensor1, train_mask_tensor1, test_mask_tensor1, parameters)
        fold_accuracies.append(accuracy)

    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)

    # Update best parameters if current combination is better
    if mean_accuracy > best_accuracy:
        best_parameters = parameters
        best_accuracy = mean_accuracy

print(f"Best parameters: {best_parameters}")
print(f"Best accuracy: {best_accuracy}")
