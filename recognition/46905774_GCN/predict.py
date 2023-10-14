import torch
import dataset
import modules
from plot import embeddings_plot


def get_model(device):
    model = modules.multi_GCN(
        dataset.sample_size, dataset.features_size, dataset.classes_size, 32
    )
    return model.to(device)


def plot_embeddings(model, filename, X, adjacency_matrix, y):
    model.eval()
    with torch.no_grad():
        out = model(X, adjacency_matrix)
    embeddings_plot(filename, out.cpu(), point_color=y.cpu())


def load_model_state(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))


def main():
    # Specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure the data is loaded onto the correct device
    X = dataset.X.to(device)
    adjacency_matrix = dataset.adjacency_matrix.to(device)
    y = dataset.y.to(device)

    # Initialize model
    model = get_model(device)

    # Plot embeddings without training
    plot_embeddings(model, "pre_train", X, adjacency_matrix, y)

    # Load trained model state and plot embeddings
    load_model_state(model, "best_model.pt", device)
    plot_embeddings(model, "post_train", X, adjacency_matrix, y)


if __name__ == "__main__":
    main()