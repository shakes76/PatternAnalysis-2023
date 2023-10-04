import torch
from modules import *
from dataset import *

def run_training(lr=0.02, num_epoch=100):
    data = load_data()
    model = GCN()
    print('\n', model, '\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion=torch.nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        optimizer.zero_grad() # clear gradients
        h, z = model(data.X, data.edges) # forward pass
        
        # disregard all test labels from criterion
        y_train = data.y
        for i in range(len(z)):
            if data.split_indices[i] == 1: # remove all test elements
                z = torch.cat([z[0:i], z[i+1:]])
                y_train = torch.cat([y_train[0:i], y_train[i+1:]])

        loss = criterion(z, y_train) # calculate loss
        acc = model.accuracy(y_train, z.argmax(dim=1)) # calculate accuracy
        loss.backward() # compute gradients
        optimizer.step() # tune parameters
        
        # print metrics every 5 epochs
        if epoch % 5 == 0:
            print(f'Epoch {epoch:>3} | Loss: {loss:.2f} | Acc: {acc*100:.2f}%')

run_training(num_epoch=50)