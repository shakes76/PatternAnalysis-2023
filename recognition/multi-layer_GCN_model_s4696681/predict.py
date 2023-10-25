import train
import torch
import dataset as dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import modules
import numpy as np
from scipy.linalg import sqrtm

def main():
    # Train the model using train.py and get the best model, get evaluation of model and the parameters of the best model
    model, best_parameters, best_accuracy = train.train_model()
    
    print(f"Best Model Parameters: {best_parameters}")
    print(f"Test accuracy with best model: {best_accuracy:.2f}%")
    torch.save(model.state_dict(), "trained_model.pth")


    # Visualisation TSNE after model is trained
    model.eval()
    with torch.no_grad():
        # Do a forward pass to compute embeddings
        _ = model(dataset.all_features_tensor, dataset.adjacency_normed_tensor)  
        embeddings = model.get_embeddings().cpu().numpy()
    
    tsne = TSNE(n_components=2, random_state=99)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    number_of_classes = 4
    
    plt.figure(figsize=(10, 8))
    for label in range(number_of_classes):
        indices = np.where(dataset.node_labels[:, 1] == label)
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=str(label), s=5)
    plt.legend()
    plt.title('t-SNE visualization of GCN embeddings')
    plt.show()
if __name__ == "__main__":
    main()
