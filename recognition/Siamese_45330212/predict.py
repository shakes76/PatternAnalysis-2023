# Shows example usage of my trained model. Print out any results and / or provide visualisations where applicable
from modules import *
from dataset import *
import time
import torch
import torch.nn as nn
import time

def save_plot(iteration, loss, epoch, title):# for showing loss value changed with iter
    plt.plot(iteration, loss)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.savefig('iteration_loss_' + str(epoch) + '.png')
    
def save_plot_acc(iteration, loss, epoch, title, xtitle, ytitle):# for showing loss value changed with iter
    plt.plot(iteration, loss)
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.savefig(str(epoch) + '.png')

def show_plot(iteration, loss, title):# for showing loss value changed with iter
    plt.plot(iteration, loss)
    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.show()

def runtest():
    classification_model = BinaryClassifier()
    classification_model.load_state_dict(torch.load("/home/Student/s4533021/classification_model.pt"))
    classification_model = classification_model.to(device)
    model = ResNet18()
    model.load_state_dict(torch.load("/home/Student/s4533021/siamese_model2.pt"))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Warning message if CUDA is not available
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")
    else:
        print("Using CUDA.")
    # --------------
    # Test the model
    print("> Testing")
    start = time.time() #time generation
    classification_model.eval()
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for img, label in test_loader:
            img, label = img.to(device) , label.to(device)
            embeddings = model.forward_once(img)
            output = classification_model(embeddings)
            _, predicted = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        if total != 0:
            print('Test Accuracy: {} %'.format(100 * correct / total))
        else:
            print('Total is 0', correct, total)
    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    print('END')