'''
Training, validation, testing and saving of ViT model on ADNI dataset.
'''
import argparse
import time

from typing import Any

import pandas as pd
import torch

from tqdm import tqdm

from dataset import BATCH_SIZE, create_train_dataloader, create_test_dataloader
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

        return self.counter >= self.patience

### TRAINING ###################################################################

def train_model(mdl: Any, epochs: int, device: torch.device, pg: bool = False) -> None:
    '''
    Train the given model on a pre-processsed ADNI training dataset.
    '''
    wrapiter = (lambda iter: tqdm(iter)) if pg else (lambda iter: iter)

    train_loader, valid_loader = create_train_dataloader(val_pct=0.2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(mdl.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    earlystop = EarlyStopping(mdl, min_delta=0.01, patience=5)

    # Training and validation metric tracking
    train_losses = []; train_acc = []
    valid_losses = []; valid_acc = []

    mdl.train()
    time_start = time.time()

    for epoch in range(epochs):
        # Training loop
        losses = []; total = 0; correct = 0
        for images, labels, _ in wrapiter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = mdl(images)
            loss = criterion(outputs, labels)
            # Record training metrics
            losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Backward and optimize
            loss.backward()
            optimizer.step()

        train_losses.append(sum(losses) / len(losses))
        train_acc.append(correct / total)

        # Validation loop
        losses = []; total = 0; correct = 0
        with torch.no_grad():
            for images, labels, _ in wrapiter(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = mdl(images)
                # Record validation metrics
                losses.append(criterion(outputs, labels).item())
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        valid_losses.append(sum(losses) / len(losses))
        valid_acc.append(correct / total)

        scheduler.step(valid_losses[-1])

        # Training report
        time_elapsed = time.time() - time_start
        print(f'Epoch [{epoch+1:02}/{epochs:02}]  Val Loss: {valid_losses[-1]:.5f}  ',
              f'Val Acc: {valid_acc[-1]:.5f}  ({strftime(time_elapsed)})')

        # Check early stopping condition
        if earlystop.stop_training(valid_losses[-1]):
            print(f'Stopping early, metric did not improve by more than '
                  f'{earlystop.min_delta} for {earlystop.patience} consecutive '
                  f'epochs')
            break

    # Export training and validation metrics
    df = pd.DataFrame(zip(*[train_losses, train_acc, valid_losses, valid_acc]),
        columns=['train_loss', 'train_acc', 'valid_loss', 'valid_acc'])
    df.to_csv(f'adni-vit-metrics-{earlystop.mdl_timestamp}.csv')

    # Return model from after epoch with lowest loss
    if earlystop.metric_best == valid_losses[-1]:
        return mdl
    else:
        return load_model(earlystop.mdl_timestamp)

### EVALUATION #################################################################

def test_model_noagg(mdl: Any, device: torch.device, pg: bool = False) -> None:
    '''
    Test the given model on the ADNI test dataset, treating each image as
    independent and not aggregating per patient.
    '''
    wrapiter = (lambda iter: tqdm(iter)) if pg else (lambda iter: iter)

    test_loader = create_test_dataloader()
    mdl.eval()
    time_start = time.time()

    total = 0; correct = 0

    # Model inference
    with torch.no_grad():
        for images, labels, _ in wrapiter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = mdl(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    time_elapsed = time.time() - time_start
    print(f'Test accuracy: {100*correct/total:.2f}% ({strftime(time_elapsed)})')

def test_model_agg(mdl: Any, device: torch.device, pg: bool = False) -> None:
    '''
    Test the given model on the ADNI test dataset, aggregating predictions
    per patient before making a final prediction.
    '''
    wrapiter = (lambda iter: tqdm(iter)) if pg else (lambda iter: iter)

    # Mapping of patient ID to prediction tallies
    prediction = {}; actual = {}

    test_loader = create_test_dataloader()
    mdl.eval()
    time_start = time.time()

    # Model inference
    with torch.no_grad():
        for images, labels, pid in wrapiter(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = mdl(images)
            _, predicted = torch.max(outputs.data, 1)

            for i in range(BATCH_SIZE):
                # Tally predictions on per-patient basis
                if pid[i] not in prediction:
                    prediction[pid[i]] = 0
                if predicted[i] == 0:
                    prediction[pid[i]] -= 1 # minus 1 from tally if NC predicted
                else:
                    prediction[pid[i]] += 1 # add 1 to tally if AD predicted

                # Record actual label, also on per-patient basis
                if pid[i] not in actual:
                    actual[pid[i]] = labels[i].item()

    # Count predictions per patient to get overall prediction
    assert prediction.keys() == actual.keys()
    total = len(prediction)
    correct = 0
    for pid, tally in prediction.items():
        pred = tally > 0 # implicitly predicts NC if AD and NC equally predicted
        correct += int(pred == actual[pid])

    time_elapsed = time.time() - time_start
    print(f'Test accuracy (agg.): {100*correct/total:.2f}% ({strftime(time_elapsed)})')

def test_model(mdl: Any, device: torch.device, agg: bool = False,
               pg: bool = False) -> None:
    '''Test the given model on the ADNI test dataset.'''
    if agg:
        test_model_agg(mdl, device, pg)
    else:
        test_model_noagg(mdl, device, pg)

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
