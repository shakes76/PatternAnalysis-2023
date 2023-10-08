import dataset
from modules import *
import torch.optim as optim
import matplotlib.pyplot as plt

def visualize_loss(batch_losses):
    
    epochs = range(1, len(batch_losses) + 1)
    plt.plot(epochs, batch_losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Loss')
    plt.title('Batch Loss Over Epochs')
    plt.grid(True)
    plt.savefig('plots/loss_plot.png')
    plt.show()


def test_accuracy(model):
    dataset = ds.ADNI_Dataset()
    test_laoder = dataset.get_test_loader()
        
    correct_predictions = 0
    total_samples = 0    
        
    model.eval() 
    for j, (images, labels) in  enumerate(test_laoder):
        if images.size(0) == 32:

            images = images.to(device) 
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = (outputs >= 0.5).squeeze().long()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy