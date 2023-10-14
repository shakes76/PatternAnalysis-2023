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

### UTILITIES ##################################################################

strftime = lambda t: f'{int(t//3600):02}:{int((t%3600)//60):02}:{(t%3600)%60:08.5f}'

def save_model(mdl: Any, timestamp: int = None) -> None:
    '''Export the given model.'''
    torch.save(mdl, f'adni-vit-trained-{timestamp or int(time.time())}.pt')

def load_model(timestamp: int) -> Any:
    '''Import the model saved with the given timestamp.'''
    return torch.load(f'adni-vit-trained-{timestamp}.pt')

class EarlyStopping:
    '''Stop training when a monitored metric has stopped improving.'''
    def __init__(self, mdl: Any, mode: str = 'min', min_delta: float = 0.0,
                 patience: int = 0) -> None:
        self.mdl = mdl
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.counter = 0
        self.metric_best = {'min': float('inf'), 'max': 0}[mode]
        self.mdl_timestamp = int(time.time())

    def stop_training(self, metric: float) -> bool:
        '''Returns true once the monitored metric has stopped improving.'''
        if self.mode == 'min':
            improved = metric < (self.metric_best - self.min_delta)
        elif self.mode == 'max':
            improved = metric > (self.metric_best + self.min_delta)
        else:
            raise ValueError('unrecognised early stopping mode')

        if improved:
            self.metric_best = metric
            self.counter = 0
            save_model(self.mdl, self.mdl_timestamp)
        else:
            self.counter += 1

        return self.counter > self.patience

### TRAINING ###################################################################

def train_model(mdl: Any, epochs: int, device: torch.device, pg: bool = False) -> None:
    '''
    Train the given model on a pre-processsed ADNI training dataset.
    '''
    wrapiter = (lambda iter: tqdm(iter)) if pg else (lambda iter: iter)

    train_loader, valid_loader = create_train_dataloader(val_pct=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(mdl.parameters(), lr=1e-4)
    earlystop = EarlyStopping(mdl, min_delta=1.0, patience=3)

    # TODO: lr scheduler

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
            losses = []
            for images, labels in wrapiter(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = mdl(images)
                losses.append(criterion(outputs, labels))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Training report
        time_elapsed = time.time() - time_start
        loss_average = sum(losses) / len(losses)
        print(f'Epoch [{epoch+1:02}/{epochs:02}]  Val Loss: {loss_average:.5f}  ',
              f'Val Acc: {correct/total:.5f}  ({strftime(time_elapsed)})')

        # Check early stopping condition
        if earlystop.stop_training(loss_average):
            print(f'Stopping early, metric did not improve by more than '
                  f'{earlystop.min_delta} for {earlystop.patience} consecutive '
                  f'epochs')
            break

    # Return model from after epoch with lowest loss
    if earlystop.metric_best == loss_average:
        return mdl
    else:
        return load_model(earlystop.mdl_timestamp)

### EVALUATION #################################################################

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

### ENTRYPOINT #################################################################

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mdl = ViT().to(device)
    mdl = train_model(mdl, epochs=args.epochs, device=device, pg=args.pg)
    test_model(mdl, device=device, pg=args.pg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('--pg', action='store_true', help='show progress bar')
    args = parser.parse_args()
    main(args)
