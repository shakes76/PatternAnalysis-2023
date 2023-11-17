import torch
import time
import matplotlib.pyplot as plt
from modules import *
from dataset import *

def run_training(lr, num_epochs):
    data = load_data()
    model = GCN(data.train_split, data.test_split)
    print('\n', model, '\n')

    accuracies = []
    losses = []

    start = time.time() # we time how long training takes

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        optimizer.zero_grad() # clear gradients
        h, z = model(data.X, data.edges) # forward pass
        
        # keep only training elements
        y_train = data.y[data.train_split]
        z = z[data.train_split]
        
        loss = criterion(z, y_train) # calculate loss
        accuracy = model.accuracy(y_train, z.argmax(dim=1)) # calculate accuracy
        loss.backward() # compute gradients
        optimizer.step() # tune parameters

        # store metrics for graphing
        accuracies.append(accuracy.item())
        losses.append(loss.item())
        
        # print metrics every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch + 1:>3} | Loss: {loss:.2f} | Acc: {accuracy*100:.2f}%')

    end = time.time()
    print('time spent training:', end - start)
    torch.save(model, 'model.pth')

    # plot accuracy
    fig, (ax0, ax1) = plt.subplots(2)
    ax0.plot([i + 1 for i in range(num_epochs)], accuracies, c='blue')
    ax0.set_title('Training accuracy')
    ax0.set(xlabel='Epoch', ylabel='Accuracy')
    ax0.grid(alpha=0.4)

    # plot loss
    ax1.plot([i + 1 for i in range(num_epochs)], losses, c='green')
    ax1.set_title('Training loss')
    ax1.set(xlabel='Epoch', ylabel='Loss')
    ax1.grid(alpha=0.4)

    fig.tight_layout(pad=2)
    plt.show()

run_training(lr=0.0175, num_epochs=100)