'''
@file   train.py
@brief  Contains the source code for training, validating, testing and saving the vision
        transformer model. The script uses the modules from modules.py and the dataloader(s)
        from dataset.py to train. The script will plot the losses and accuracy of the training
        and validation sets 
@date   20/10/2023
'''

import torch
import torch.nn as nn
from torch import optim
from torch.hub import tqdm
from dataset import DataLoader
from modules import ViT
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import sys

'''
HyperParameters

This class stores all the hyperparameters for the vision transformer
'''
class HyperParameters(object):
    def __init__(self) -> None:
        self.patch_size = 8             # image patch size
        self.mlp_dim = 128              # dimension of mlp in transformer encoder
        self.head_dim = 512             # dimension of mlp head
        self.n_channels = 3             # number of channels for convultional layer
        self.num_encoders = 4           # number of transformer encoders
        self.num_heads = 4              # number of attention heads
        self.dropout = 0.0              # dropout regularisation
        self.num_classes = 2            # number of classes
        self.epochs = 60                # max number of epochs
        self.lr = 1e-3                  # learning rate
        self.weight_decay = 0.00        # weight decay regularisation
        self.batch_size = 32            # batch size
        self.hidden_size = 64           # size of convolutional layer

'''
Check the accuracy of a model on the given dataloader
Parameters:
    loader - data loader
    model - deep learning model object
    device - device to run model on (cuda, cpu)
'''
def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Get predictions
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)

    model.train()
    accuracy = num_correct / num_samples    # accuracy of model on dataloader
    return accuracy

'''
Train a specified model on the train loader using the given hyperparameters
Parameters:
    args - hyperparameter object
    train_loader - train dataset dataloader object
    epoch - the current epoch of training
    device - device to run model on
    model - deep learning model to be trained
    criterion - loss criterion i.e. BCELoss
    optimizer - training optimizer i.e. Adam/Adagrad
'''
def train_model(args, train_loader, epoch, device, model, criterion, optimizer):
    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    tk = tqdm(train_loader, desc="EPOCH" + "[TRAIN]" 
                + str(epoch + 1) + "/" + str(args.epochs))
    
    for batch_idx, (data, targets) in enumerate(tk):
        # Get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)

        # Forward propogation
        scores = model(data)
        _, predictions = scores.max(1)
        num_correct += (predictions == targets).sum()
        num_samples += predictions.size(0)
        loss = criterion(scores, targets)

        # Back propogation
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        optimizer.step()

        total_loss += loss.item()
        tk.set_postfix({"Loss": "%6f" % float(total_loss / (batch_idx + 1))})
    
    accuracy = num_correct / num_samples
    return ((total_loss / len(train_loader)), accuracy)

'''
Use a validation dataset to validate the model on
Parameters:
    args - HyperParameters object
    val_loader - validation dataset dataloader
    model - deep learning model to be validated
    epoch - current epoch of training
    device - device to run model on
    criterion - loss criterion
'''
def validate_model(args, val_loader, model, epoch, device, criterion):
    model.eval()
    total_loss = 0.0
    num_correct = 0
    num_samples = 0
    tk = tqdm(val_loader, desc="EPOCH" + "[VALID]" + str(epoch + 1) + "/" + str(args.epochs))

    # No gradient
    with torch.no_grad():
        for t, (data, targets) in enumerate(tk):
            data, targets = data.to(device), targets.to(device)

            # Get predictinos
            scores = model(data)
            _, predictions = scores.max(1)

            # Calculate numbre of correct scores
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
            loss = criterion(scores, targets) # Get loss

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (t + 1))})

    accuracy = num_correct / num_samples
    print(f"Accuracy on validation set: {accuracy *100:.2f}")
    model.train()
    return ((total_loss / len(val_loader)), accuracy)

'''
Plot the losses of training
Parameters:
    train_losses - losses of training dataset at each epoch
    val_losses - losses of validation dataset at each epoch
'''
def plot_losses(train_losses, val_losses):
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Training and Validation Set Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend(["Training", "Validation"])
    plt.show()

'''
Plot the accuracy of training
Parameters:
    train_accuracy - accuracy of training dataset at each epoch
    val_accuracy - accuracy of validation dataset at each epoch
'''
def plot_accuracy(train_accuracy, val_accuracy):
    plt.plot(train_accuracy)
    plt.plot(val_accuracy)
    plt.title('Training and Validation Set Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(["Training", "Validation"])
    plt.show()

def main():

    # Check if we need to plot losses and accuracy
    plotLossAccuracy = True
    for i in range(len(sys.argv)):
        if sys.argv[i] =='--hide-plots':
            if sys.argv[i + 1] == 'true':
                plotLossAccuracy = False
            else:
                plotLossAccuracy = True

    args = HyperParameters() # Hyperparameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Get dataloaders
    dl = DataLoader(batch_size=args.batch_size) # Dataloader object from dataset.py
    train_loader = dl.get_training_loader()
    test_loader = dl.get_test_loader()
    valid_loader = dl.get_valid_loader()

    # Define variables for training
    trainLoss = 0.0
    trainAccuracy = 0.0
    trainLossList = []
    trainAccuracyList = []
    validLoss = 0.0
    validAccuracy = 0.0
    validLossList = []
    validAccuracyList = []

    # Define the vision transformer model
    model = ViT(args).to(device)

    # Use Adam optimizer, CE loss, and lr schedular to reduce on plateau
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8, min_lr=1e-8, 
                                               verbose=True)
    model.train()

    # Iterate through max number of epochs
    for epoch in range(args.epochs):
        # Train the model 
        trainLoss, trainAccuracy = train_model(args, train_loader, epoch, device, model, criterion, optimizer)

        # Validate the model 
        validLoss, validAccuracy = validate_model(args, valid_loader, model, epoch, device, criterion)

        # Adjust the lr appropriately
        scheduler.step(trainLoss)

        trainLossList.append(trainLoss)
        validLossList.append(validLoss)
        trainAccuracyList.append(float(trainAccuracy))
        validAccuracyList.append(float(validAccuracy))
    
    # Get test dataset accuracy
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")
    
    # Plot losses and accuracy of training and validation
    if plotLossAccuracy:
        plot_losses(trainLossList, validLossList)
        plot_accuracy(trainAccuracyList, validAccuracyList)

    # Save the model
    torch.save(model, 'model.pt')

if __name__ == "__main__":
    main()