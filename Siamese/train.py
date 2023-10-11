import torch
import torch.optim as optim
from modules import SiameseNetwork, ContrastiveLoss
from dataset import get_dataloader

def train_epoch(model, dataloader, criterion, optimizer):
    # TODO: Implement training logic for one epoch
    pass

def validate_epoch(model, dataloader, criterion):
    # TODO: Implement validation logic for one epoch
    pass

def main():
    # Initialize model, criterion, and optimizer
    model = SiameseNetwork()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters())

    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()

    # TODO: Training loop
    pass

if __name__ == '__main__':
    main()