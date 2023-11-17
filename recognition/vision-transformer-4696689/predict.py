"""
Imports
"""
import torch
from train import test
from dataset import trainloader, valloader, testloader
from numpy import loadtxt
import matplotlib.pyplot as plt

model = torch.jit.load('model_trained.pt')
model.eval()

#try load and plot loss curve
try:
    loss = loadtxt('loss.txt')
    steps = len(loss)
    plt.plot(steps, LOSS)
    plt.ylabel('LOSS')
    plt.xlabel('epoch')
    plt.title('Training Loss')
    plt.show()
except:
    print("No training loss!")

#try load and plot accuracy curve
try:
    loss = loadtxt('acc.txt')
    steps = len(loss)
    plt.plot(steps, LOSS)
    plt.ylabel('ACCURACY')
    plt.xlabel('epoch')
    plt.title('Validation Accuracy')
    plt.show()
except:
    print("No accuracy!")
    
#try load and plot train accuracy curve
try:
    loss = loadtxt('train.txt')
    steps = len(loss)
    plt.plot(steps, LOSS)
    plt.ylabel('ACCURACY')
    plt.xlabel('epoch')
    plt.title('Training Accuracy')
    plt.show()
except:
    print("No training accuracy")

"""train models on datasets"""
# train_acc = test(model, trainloader) #test on train set
# val_acc = test(model, valloader) #test on validation set
# test_acc = test(model, testloader) #test on test set

# print("accuracy on training set:", train_acc)
# print("accuracy on validation set:", val_acc)
# print("accuracy on test set:", test_acc)