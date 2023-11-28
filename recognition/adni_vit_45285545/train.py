'''
Training, validation, testing and saving of ViT model on ADNI dataset.
'''
import argparse
import time
from typing import Any, Iterable, Tuple

import pandas as pd
import tqdm

import torch
from torchvision.models.vision_transformer import ViT_B_16_Weights

from dataset import create_train_dataloader, create_test_dataloader
from modules import ViT

### UTILITIES ##################################################################

# Format a Unixtime timestamp into hh:mm:ss.sssss
strftime = lambda t: f'{int(t//3600):02}:{int((t%3600)//60):02}:{(t%3600)%60:08.5f}'

def save_model(mdl: Any, timestamp: int = None) -> None:
    '''Export the given model to a file stamped with the current Unixtime.'''
    torch.save(mdl, f'adni-vit-trained-{timestamp or int(time.time())}.pt')

def load_model(timestamp: int) -> Any:
    '''Import the model saved with the given Unixtime timestamp.'''
    return torch.load(f'adni-vit-trained-{timestamp}.pt')

class EarlyStopping:
    '''
    Suggest to stop training when a monitored metric has stopped improving.

    Args:
        mdl (any): Model to monitor and save each best result of, re-loading
            state after best epoch if early stopping triggers.
        mode (str): 'min' or 'max'. If 'min', model improves when the monitored
            metric decreases. If 'max', model improves when the monitored metric
            increases.
        min_delta (float): minimum absolute improvement in monitored metric for
            model to be considered as improved.
        patience (int): number of consecutive non-improved epochs before early
            stopping returns True (doesn't actually stop the training itself).

    '''

    def __init__(
        self,
        mdl: Any,
        mode: str = 'min',
        min_delta: float = 0.0,
        patience: int = 0
    ) -> None:

        self.mdl = mdl
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience

        # Keep a counter of the number of consecutive non-improved epochs
        self.counter = 0

        # Maintain record of best metric result
        self.metric_best = {'min': float('inf'), 'max': 0}[mode]

        # Timestamp used for auto-saving and reloading best model
        self.mdl_timestamp = int(time.time())

    def stop_training(self, metric: float) -> bool:
        '''Returns True once the monitored metric has stopped improving.'''

        # Determine if current metric counts as "improvement" over previous best
        if self.mode == 'min':
            improved = metric < (self.metric_best - self.min_delta)
        elif self.mode == 'max':
            improved = metric > (self.metric_best + self.min_delta)
        else:
            raise ValueError('unrecognised early stopping mode')

        # If model improved, save the improved model (overwriting previous save)
        if improved:
            self.metric_best = metric
            self.counter = 0            # Reset counter to 0
            save_model(self.mdl, self.mdl_timestamp)

        # Otherwise increment the counter of consecutive non-improved epochs
        else:
            self.counter += 1

        # If the number of consecutive non-improved epochs is greater than the
        # early stopping patience, returns True, suggesting early stop.
        return self.counter >= self.patience

### TRAINING ###################################################################

def train_epoch(
    mdl: Any,
    device: torch.device,
    train_loader: Iterable,
    valid_loader: Iterable,
    criterion: Any,
    optimizer: torch.optim.Optimizer,
    pg: bool = False,
) -> Tuple[float, float, float, float]:
    '''
    Trains the model for one epoch using the given training DataLoader, then
    validates the model on the given validation DataLoader.

    Args:
        mdl (any): Model to train and evaluate.
        device (torch.device): Device on which to train and evaluate model.
        train_loader (iterable): DataLoader or equivalent iterable containing
            training data.
        valid_loader (iterable): DataLoader or equivalent iterable containing
            validation data.
        criterion (any): Loss function used to evaluate model performance.
        optimizer (torch.optim.Optimizer): Optimizer used to stochastically
            train the model.
        pg (bool): If True, ``tqdm`` progress bars are shown for training and
            validation loops. Otherwise no progress bars are shown.

    Returns:
        Tuple of metrics as: (train loss, train acc, valid loss, valid acc)
    '''
    # Show tqdm progress bar or not
    irange = tqdm.trange if pg else range

    # Training loop
    losses = []; total = 0; correct = 0
    for images, labels, _ in irange(train_loader):
        # Move samples onto target device
        images = images.to(device)
        labels = labels.to(device)
        # Reset/initialise optimizer gradients
        optimizer.zero_grad()
        # Training forward pass and loss calculation
        outputs = mdl(images)
        loss = criterion(outputs, labels)
        # Record training metrics (reported as per-epoch average only)
        losses.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Training backward pass (backprop) and stochastic optimisation
        loss.backward()
        optimizer.step()

    # Calculate epoch-average model training metrics
    train_loss = sum(losses) / len(losses)
    train_acc = correct / total

    # Validation loop
    losses = []; total = 0; correct = 0
    with torch.no_grad():
        for images, labels, _ in irange(valid_loader):
            # Move samples onto target device
            images = images.to(device)
            labels = labels.to(device)
            # Run model inference on batch
            outputs = mdl(images)
            # Record validation metrics (reported as per-epoch average only)
            losses.append(criterion(outputs, labels).item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate epoch-average model validation metrics
    val_loss = sum(losses) / len(losses)
    val_acc = correct / total

    return train_loss, train_acc, val_loss, val_acc

def train_model(
    mdl: Any,
    epochs: int,
    device: torch.device,
    pg: bool = False
) -> None:
    '''
    Trains the given model on a pre-processsed ADNI training + validation set,
    with a validation split of 20%. Exports per-epoch training and validation
    metrics to a Unixtime timestamped CSV file, and saves the model after the
    best epoch to another file with the same timestamp.

    Args:
        mdl (any): Model to train and evaluate.
        epochs (int): Maximum number of epochs to train model (will stop early
            if model is not improving).
        device (torch.device): Device on which to train and evaluate model.
        pg (bool): If True, ``tqdm`` progress bars are shown for training and
            validation loops. Otherwise no progress bars are shown. Defaults
            to False.

    '''
    # Create training and validation data loaders with 80/20 split
    train_loader, valid_loader = create_train_dataloader(val_pct=0.2)

    # Initialise loss criterion, optimiser, LR scheduler and early stopping.
    # Each of their parameters represent tunable hyperparameters.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RAdam(mdl.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
    earlystop = EarlyStopping(mdl, min_delta=0.01, patience=5)

    # Lists for keeping track of training and validation metrics
    train_losses = []; train_acc = []
    valid_losses = []; valid_acc = []

    mdl.train()                 # set model to training mode
    time_start = time.time()    # start training timer

    # Freeze non-head layers (for fine-tuning ImageNet1K pre-trained model)
    for param in mdl.parameters():
        param.requires_grad = False
    for param in mdl.heads.parameters():
        param.requires_grad = True

    # Train for 1 epoch with non-head layers frozen
    metrics = train_epoch(
        mdl, device, train_loader, valid_loader, criterion, optimizer, pg)

    train_losses.append(metrics[0]); train_acc.append(metrics[1])
    valid_losses.append(metrics[2]); valid_acc.append(metrics[3])

    # Print frozen epoch training report
    time_elapsed = time.time() - time_start
    print(f'Epoch [00/{epochs:02}]  Val Loss: {valid_losses[-1]:.5f}  ',
            f'Val Acc: {valid_acc[-1]:.5f}  ({strftime(time_elapsed)})')

    # Un-freeze all layers
    for param in mdl.parameters():
        param.requires_grad = True

    # Train for specified number of epochs with all layers unfrozen
    for epoch in range(epochs):
        metrics = train_epoch(
            mdl, device, train_loader, valid_loader, criterion, optimizer, pg)

        train_losses.append(metrics[0]); train_acc.append(metrics[1])
        valid_losses.append(metrics[2]); valid_acc.append(metrics[3])

        scheduler.step(valid_losses[-1]) # update LR scheduler

        # Print epoch training report
        time_elapsed = time.time() - time_start
        print(f'Epoch [{epoch+1:02}/{epochs:02}]  Val Loss: {valid_losses[-1]:.5f}  ',
              f'Val Acc: {valid_acc[-1]:.5f}  ({strftime(time_elapsed)})')

        # Check early stopping condition; we choose to monitor validation loss
        if earlystop.stop_training(valid_losses[-1]):
            print(f'Stopping early, metric did not improve by more than '
                  f'{earlystop.min_delta} for {earlystop.patience} consecutive '
                  f'epochs')
            break

    # Export training and validation metrics to Unixtime timestamped CSV
    df = pd.DataFrame(zip(*[train_losses, train_acc, valid_losses, valid_acc]),
        columns=['train_loss', 'train_acc', 'valid_loss', 'valid_acc'])
    df.to_csv(f'adni-vit-metrics-{earlystop.mdl_timestamp}.csv')

    # Return model from after epoch with best metric (lowest validation loss)
    if earlystop.metric_best == valid_losses[-1]:
        return mdl
    else:
        return load_model(earlystop.mdl_timestamp)

### EVALUATION #################################################################

def test_model_noagg(mdl: Any, device: torch.device, pg: bool = False) -> None:
    '''
    Test the given model on the ADNI test dataset, treating each image as
    independent and not aggregating per patient.

    Args:
        mdl (any): Model to evaluate on ADNI test split.
        device (torch.device): Device on which to test the model.
        pg (bool): If True, ``tqdm`` progress bar is shown for testing loop.
            Otherwise no progress bar is shown. Defaults to False.

    '''
    # Show tqdm progress bar or not
    wrapiter = (lambda iter: tqdm.tqdm(iter)) if pg else (lambda iter: iter)

    # Create test data loader
    test_loader = create_test_dataloader()

    mdl.eval()                  # set model to test mode
    time_start = time.time()    # start test timer

    # Keep track of testing metrics
    total = 0; correct = 0

    # Run model inference on test set
    with torch.no_grad():
        for images, labels, _ in wrapiter(test_loader):
            # Move samples onto target device
            images = images.to(device)
            labels = labels.to(device)
            # Run model inference on batch
            outputs = mdl(images)
            # Record batch testing metrics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print test report
    time_elapsed = time.time() - time_start
    print(f'Test accuracy: {100*correct/total:.2f}% ({strftime(time_elapsed)})')

def test_model_agg(mdl: Any, device: torch.device, pg: bool = False) -> None:
    '''
    Test the given model on the ADNI test dataset, aggregating predictions
    per patient before making a final prediction.

    Args:
        mdl (any): Model to evaluate on ADNI test split.
        device (torch.device): Device on which to test the model.
        pg (bool): If True, ``tqdm`` progress bar is shown for testing loop.
            Otherwise no progress bar is shown. Defaults to False.

    '''
    # Show tqdm progress bar or not
    wrapiter = (lambda iter: tqdm.tqdm(iter)) if pg else (lambda iter: iter)

    # Mapping of patient ID to prediction tallies
    prediction = {}; actual = {}

    # Create test data loader
    test_loader = create_test_dataloader()

    mdl.eval()                  # set model to test mode
    time_start = time.time()    # start test timer

    # Run model inference on test set
    with torch.no_grad():
        for images, labels, pid in wrapiter(test_loader):
            # Move samples onto target device
            images = images.to(device)
            labels = labels.to(device)
            # Run model inference on batch
            outputs = mdl(images)
            _, predicted = torch.max(outputs.data, 1)

            # For each patient ID in the test set, count the number of positive
            # vs negative predictions to determine overall prediction
            for i in range(len(pid)):
                k_pid = pid[i].item()
                # Tally predictions on per-patient basis
                if k_pid not in prediction:
                    prediction[k_pid] = 0
                if predicted[i] == 0:
                    prediction[k_pid] -= 1 # minus 1 from tally if NC predicted
                else:
                    prediction[k_pid] += 1 # add 1 to tally if AD predicted

                # Record actual label, also on per-patient basis
                if k_pid not in actual:
                    actual[k_pid] = labels[i].item()

    # Count predictions per patient to get overall prediction
    assert prediction.keys() == actual.keys()
    total = len(prediction)
    correct = 0
    for pid, tally in prediction.items():
        pred = tally > 0 # implicitly predicts NC if AD and NC equally predicted
        correct += int(pred == actual[pid])

    # Print test report
    time_elapsed = time.time() - time_start
    print(f'Test accuracy (agg.): {100*correct/total:.2f}% ({strftime(time_elapsed)})')

def test_model(
    mdl: Any,
    device: torch.device,
    agg: bool = False,
    pg: bool = False
) -> None:
    '''
    Test the given model on the ADNI test dataset. Testing can happen two ways:
    1. Aggregating predictions per-patient to give overall prediction; or
    2. Treating each sample as independent in scoring prediction accuracy.

    Args:
        mdl (any): Model to evaluate on ADNI test split.
        device (torch.device): Device on which to test the model.
        agg (bool): If True, aggregates prediction on per-patient basis for
            scoring. Otherwise scores each prediction as independent samples.
            Defaults to False.
        pg (bool): If True, ``tqdm`` progress bar is shown for testing loop.
            Otherwise no progress bar is shown. Defaults to False.

    '''
    if agg:
        test_model_agg(mdl, device, pg)
    else:
        test_model_noagg(mdl, device, pg)

### ENTRYPOINT #################################################################

def main(args):
    # Default to CUDA device if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model initialised with num_classes=1000 so ImageNet weights can be loaded.
    # Chosen params. identical to original ViT-B/16 (Dosovitsky et al, 2021).
    mdl = ViT(224, 16, 12, 12, 768, 3072, num_classes=1000)
    weights = ViT_B_16_Weights.IMAGENET1K_V1 # ImageNet pre-trained weights
    mdl.load_state_dict(weights.get_state_dict(progress=True, check_hash=True))

    # Now replace model head with equivalent 2-output classifier
    mdl.heads.head = torch.nn.Linear(768, 2)
    torch.nn.init.zeros_(mdl.heads.head.weight) # must initialise manually as
    torch.nn.init.zeros_(mdl.heads.head.bias)   # new layers are uninitialised

    # Train the model, returning state after epoch with lowest validation loss.
    # Training and validation metrics are exported to timestamped CSV file, and
    # best model is also saved with timestamp.
    mdl = mdl.to(device)
    mdl = train_model(mdl, epochs=args.epochs, device=device, pg=args.pg)

    # Test the model and print a test report
    test_model(mdl, device=device, pg=args.pg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int, help='number of training epochs')
    parser.add_argument('--pg', action='store_true', help='show progress bar')
    args = parser.parse_args()
    main(args)
