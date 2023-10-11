'''
Training, validation, testing and saving of ViT model on ADNI dataset.
'''
import argparse
import time

from typing import Any

import torch

from tqdm import tqdm

from dataset import create_train_dataloader, create_test_dataloader
from modules import ViT

N_EPOCHS = 20

strftime = lambda t: f'{int(t//3600):02}:{int((t%3600)//60):02}:{(t%3600)%60:08.5f}'

def train_model(mdl: Any, epochs: int, device: torch.device, pg: bool = False) -> None:
    '''
    Train the given model on a pre-processsed ADNI training dataset.
    '''
    wrapiter = (lambda iter: tqdm(iter)) if pg else (lambda iter: iter)

    train_loader, valid_loader = create_train_dataloader(val_pct=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(mdl.parameters(), lr=1e-4)
    mdl.train()
    time_start = time.time()
    for epoch in range(epochs):
        # Training loop
        for images, labels in wrapiter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = mdl(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()

        # Validation loop
        with torch.no_grad():
            total = 0
            correct = 0
            for images, labels in wrapiter(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = mdl(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Training report
        time_elapsed = time.time() - time_start
        print(f'Epoch [{epoch+1:02}/{epochs:02}]  Loss: {loss.item():.5f}  ',
              f'Val Acc: {correct/total:.5f}  ({strftime(time_elapsed)})')

def test_model(mdl: Any, device: torch.device, pg: bool = False) -> None:
    '''Test the given model on the ADNI test dataset.'''
    wrapiter = (lambda iter: tqdm(iter)) if pg else (lambda iter: iter)

    test_loader = create_test_dataloader()
    mdl.eval()
    time_start = time.time()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in wrapiter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = mdl(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        time_elapsed = time.time() - time_start
        print(f'Test accuracy: {100*correct/total:.2f}% ({strftime(time_elapsed)})')

def save_model(mdl: Any) -> None:
    '''Export the given model.'''
    timestamp = int(time.time())
    torch.save(mdl, f'adni-vit-trained-{timestamp}')

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = ViT().to(device)
    train_model(mdl, epochs=args.epochs, device=device, pg=args.pg)
    test_model(mdl, device=device, pg=args.pg)
    save_model(mdl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('--pg', action='store_true', help='show progress bar')
    args = parser.parse_args()
    main(args)
