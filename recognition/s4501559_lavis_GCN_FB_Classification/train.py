"""
Train Model (train.py)

Script to train the GCN model and output the result. Model is trained here using cross entropy loss and stochastic gradient decent.

Author: James Lavis (s4501559)
"""

from dataset import Dataset
from modules import GCN
import torch
from tqdm import tqdm
import pickle
from torch.optim.lr_scheduler import ExponentialLR

def train_model(model, optimizer, loss_func, dataloader, lr_schedular = None, epochs=400):
    """
    Given a model, optimizer, dataloader and a loss function, train the model.
    """
    epochs = tqdm(range(epochs))
    for epoch in epochs:
        model.train()
        optimizer.zero_grad()
        loss = 0
        for batch in dataloader:
            pred = model(batch)
            loss += loss_func(pred[batch.train_mask], batch.y[batch.train_mask])
        loss /= len(dataloader)
        loss.backward()
        optimizer.step()
        if lr_schedular:
            lr_schedular.step()

        epochs.set_description(f"{epoch}/{len(epochs)}, Loss {loss:.4f}")


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"> Device = {device}")

    data = Dataset('datasets/facebook.npz', device=device)
    print("> Data successfully loaded")

    dataloader = data.data_loader(batchsize=32)
    gcn_model = GCN(data.graph.x.shape[1], 4, hidden_layers=[128]*2).to(device)
    optimizer = torch.optim.SGD(gcn_model.parameters(), lr=0.65)
    # lr_schedular = ExponentialLR(optimizer, gamma=0.1)
    CELoss = torch.nn.CrossEntropyLoss()

    print("> Training start")
    train_model(model=gcn_model, optimizer=optimizer, loss_func=CELoss, dataloader=dataloader, lr_schedular=None)

    print("> Saving model to picle")
    with open("models/gcn_model.pkl", "wb") as file:
        pickle.dump(gcn_model, file)
        file.close()
    print("> Model Saved Sucessfully")