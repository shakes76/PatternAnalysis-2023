'''
@file   predict.py
@brief  Shows an example usage of the trained vision transformer model. The script picks a random batch
        from the test dataset, displays this batch, and predicts the class for each image. The accuracy
        and cross entropy loss is then calculated for the batch.
@date   20/10/2023
'''

import torch
from dataset import DataLoader
import matplotlib.pyplot as plt
import random
import torch.nn as nn

'''
Check the accuracy of the model on the given dataloader at the specified batch number
Parameters:
    loader - PyTorch DataLoader object
    model - deep learning model
    device - device to run the model on (e.g. cuda/cpu)
    batch_number - the batch to check the accuracy of
    dl - DataLoader object from dataset.py
    criterion - loss criterion
'''
def check_accuracy(loader, model, device, batch_number, dl, criterion):
    num_correct = 0
    num_samples = 0
    loss = 0.0
    model.eval()

    classDict = ['AD', 'NC']

    # No gradient
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loader):
            if batch_idx == batch_number:

                # Show the batch
                dl.show_images(data, nmax=32)
                plt.show()

                # Move data and targets to device
                data = data.to(device=device)
                targets = targets.to(device=device)

                # Predict the scores, calculate accuracy and CE loss
                scores = model(data)
                _, predictions = scores.max(1)
                num_correct += (predictions == targets).sum()
                num_samples += predictions.size(0)
                loss = criterion(scores, targets)

                # Print the results
                for i in range(num_samples):
                    print(f'Sample {i}:\r\nTarget Class {classDict[int(targets[i])]} - Predicted Class {classDict[int(predictions[i])]}\r\n')
            elif batch_idx > batch_number:
                model.train()
                return ((num_correct / num_samples), loss)
            else:
                continue

    model.train()
    accuracy = num_correct / num_samples
    return (accuracy, loss)

def main():

    # Get cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model in
    model = torch.load('model.pt')
    model.eval()

    # Get the test dataset 
    dl = DataLoader(batch_size=32)
    testloader = dl.get_test_loader()

    # Pick a random batch between 0 and 15
    idx = random.randint(0, 15)
    print(f'Using batch {idx}\r\n')

    # Predict
    accuracy, loss = check_accuracy(testloader, model, device, idx, dl, nn.CrossEntropyLoss())
    print(f'Test Accuracy {accuracy * 100:.2f}')
    print(f'Test CE Loss {loss:.2f}')

if __name__ == '__main__':
    main()