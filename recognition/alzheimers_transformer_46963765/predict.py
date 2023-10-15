import dataset as ds
from modules import *
import matplotlib.pyplot as plt

def visualize_loss(batch_losses):
    # epochs on the x axis, batch loss on y axis 
    epochs = range(1, len(batch_losses) + 1)
    plt.plot(epochs, batch_losses, marker='o', linestyle='-')
    
    # apply labels
    plt.xlabel('Epoch')
    plt.ylabel('Batch Loss')
    plt.title('Batch Loss Over Epochs')
    plt.grid(True)
    #save plot and show
    plt.savefig('plots/loss_plot.png')
    plt.show()


def test_accuracy(model, batch_size):
    # load the test dataset
    dataset = ds.ADNI_Dataset()
    test_laoder = dataset.get_test_loader()
        
    correct_predictions = 0
    total_samples = 0    
        
    # run through dataset and find accuracy
    model.eval() 
    for j, (images, labels) in  enumerate(test_laoder):
        if images.size(0) == batch_size:
            # forward pass through model
            images = images.to(device); labels = labels.to(device)
            outputs = model(images).squeeze()
            # calculate total correct 
            predictions = (outputs >= 0.5).long()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # calculate final accuracy and return as percent
    accuracy = correct_predictions / total_samples
    return accuracy*100


def visualize_accuracies(accuracies):
    # epochs on the x axis, accuracies on y axis 
    epochs = range(1, len(accuracies) + 1)
    plt.plot(epochs, accuracies, marker='o', linestyle='-')
    
    #apply labels
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.grid(True)
    
    # save plot
    plt.savefig('plots/accuracy.png')

