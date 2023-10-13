"""
predict On Model (predict.py)

Script to run inference on the graph, and generate the TSNE plot

Author: James Lavis (s4501559)
"""

from dataset import Dataset
from modules import GCN
import torch
from tqdm import tqdm
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def model_inference(model, graph):
    """
    Given a model, and a graph, run inference on the graph.
    """
    pred_proba = model(graph)
    predicted_class = pred_proba.argmax(axis=1)
    return predicted_class

def tsne_plot(model, graph, savefile = '', correct = [], show=False):
    """ Extract graph embeddings from the model and generate tsne plot """
    graph_embeddings = model.embeddings(graph).to('cpu').detach().numpy()
    y = graph.y.to('cpu').detach().numpy()

    tsne = TSNE(n_components=2, perplexity=50, n_jobs=-1, early_exaggeration=24, n_iter=2000, n_iter_without_progress=500)#, learning_rate=100)
    X_tsne = tsne.fit_transform(graph_embeddings)

    if not correct:
        correct = np.ones(len(y)) == 1

    plt.figure(figsize=(10, 10))
    for i in range(4):
        filt = y == i
        filt = np.logical_and(filt, correct)
        plt.scatter(X_tsne[filt, 0], X_tsne[filt, 1])
    
    filt = ~correct
    plt.scatter(X_tsne[filt, 0], X_tsne[filt, 1], marker='.', s=40)
    plt.xlabel("tsne 1")
    plt.ylabel("tsne 2")
    plt.legend([0, 1, 2, 3, "Incorrect"])
    if savefile:    
        plt.savefig(savefile)

    if show:
        plt.show()

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"> Device = {device}")

    data = Dataset('datasets/facebook.npz', device=device)
    print("> Data successfully loaded")

    print("> Importing model from pickle")
    with open("models/gcn_model.pkl", "rb") as file:
        gcn_model = pickle.load(file)
        file.close()
    print("> Model Imported Sucessfully")

    print("> Running inference on graph")
    prediction = model_inference(gcn_model, data.graph)

    print("> Inference Complete")
    accuracy = (prediction == data.graph.y).sum()/len(data.graph.y)
    test_accuracy = (prediction[data.graph.test_mask] == data.graph.y[data.graph.test_mask]).sum()/len(data.graph.y[data.graph.test_mask])

    print("> Model Results")
    print(f"\tTest Accuracy: {test_accuracy}")
    print(f"\tFull graph Accuracy: {accuracy}")

    for i in data.graph.y.unique():
        filt = (data.graph.test_mask) & (data.graph.y == i)
        class_acc = (prediction[filt] == data.graph.y[filt]).sum()/len(data.graph.y[filt])
        print(f"Class Accuracy for class {i.to('cpu').detach().numpy()}: {class_acc.to('cpu').detach().numpy()}")

    print("> Generating TSNE graph")
    tsne_plot(gcn_model, data.graph, savefile="figures/tsne.png", show=True)