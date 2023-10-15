import torch
from torch.nn import Linear
from modules import *
from dataset import *

def run_training(lr, num_epochs):
    data = load_data()
    model = GCN(data.train_split, data.test_split)
    print('\n', model, '\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion=torch.nn.CrossEntropyLoss()

    # data for embedding plot
    embeddings = []
    losses = []
    accuracies = []
    outputs = []

    for epoch in range(num_epochs):
        optimizer.zero_grad() # clear gradients
        h, z = model(data.X, data.edges) # forward pass
        
        # keep only training elements
        y_train = data.y[data.train_split]
        z = z[data.train_split]

        # store data for embedding plot
        if epoch == num_epochs - 1:
            model.embeddings = h
            model.outputs = z.argmax(dim=1)
        
        loss = criterion(z, y_train) # calculate loss
        acc = model.accuracy(y_train, z.argmax(dim=1)) # calculate accuracy
        loss.backward() # compute gradients
        optimizer.step() # tune parameters
        
        # print metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

    torch.save(model, 'model.pth')

run_training(lr=0.0175, num_epochs=100)