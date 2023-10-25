import modules as modules
import dataset as dataset
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold, ParameterGrid
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_vectors = dataset.create_feature_vectors()
node_labels = dataset.convert_labels()
all_features_tensor, train_labels_tensor, test_labels_tensor, train_tensor, test_tensor, test_mask = dataset.create_tensors()
adjacency_normed_tensor = torch.FloatTensor(dataset.adjacency_normed).to(device)

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
param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'epochs': [100, 200, 300],
    'hidden_features': [32, 64, 128]
}
parameter_combinations = list(ParameterGrid(param_grid))

def train_and_evaluate(model, train_tensor1, test_tensor1, train_labels_tensor1, test_labels_tensor1, train_mask_tensor1, test_mask_tensor1, parameters):
    learning_rate = parameters['learning_rate']
    epochs = parameters['epochs']
    hidden_features = parameters['hidden_features']
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(all_features_tensor, adjacency_normed_tensor)
        loss = F.nll_loss(out[train_tensor1], train_labels_tensor1)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        test_out = model(all_features_tensor, adjacency_normed_tensor)
        pred = test_out[test_mask_tensor1].argmax(dim=1)
        correct = (pred == test_labels_tensor1).sum().item()
        acc = correct / test_labels_tensor1.size(0)
        print(f"Test accuracy: {acc * 100:.2f}%")
        test_accuracy = acc*100

    return test_accuracy 

def train_model():
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True)

    best_parameters = None
    best_accuracy = -1
    best_model = None 

    for parameters in parameter_combinations:
        fold_accuracies = []

        for train_indices, test_indices in kf.split(all_features_tensor):
            model = modules.GCN(in_features=all_features_tensor.shape[1], hidden_features=parameters['hidden_features'], out_features=out_features).to(device)
            train_tensor1, test_tensor1, train_labels_tensor1, test_labels_tensor1, train_mask_tensor1, test_mask_tensor1 = create_new_tensors(train_indices, test_indices)
            accuracy = train_and_evaluate(model, train_tensor1, test_tensor1, train_labels_tensor1, test_labels_tensor1, train_mask_tensor1, test_mask_tensor1, parameters)
            fold_accuracies.append(accuracy)

        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)

        if mean_accuracy > best_accuracy:
            best_parameters = parameters
            best_accuracy = mean_accuracy
            best_model = model 

    return best_parameters, best_accuracy, best_model 


if __name__ == "__main__":
    model, best_parameters, best_accuracy = train_model()
    print(f"Best parameters: {best_parameters}")
    print(f"Best accuracy: {best_accuracy}")
