'''
Training, validation, testing and saving of ViT model on ADNI dataset.
'''
import argparse
import time

from typing import Any

import torch
import torchvision

from tqdm import tqdm

from dataset import create_train_dataloader, create_test_dataloader

N_EPOCHS = 20

strftime = lambda t: f'{int(t//3600):02}:{int((t%3600)//60):02}:{(t%3600)%60:08.5f}'

def train_model(mdl: Any, epochs: int, device: torch.device) -> None:
    '''
    Train the given model on a pre-processsed ADNI training dataset.
    '''
    train_loader = create_train_dataloader()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-3)
    mdl.train()
    time_start = time.time()
    for epoch in range(epochs):
        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            # Forward pass
            outputs = mdl(images)
            loss = criterion(outputs, labels)
            # Backward and optimize
            loss.backward()
            optimizer.step()

        # Training report
        time_elapsed = time.time() - time_start
        print(f'Epoch [{epoch+1:02}/{epochs:02}]  Loss: {loss.item():.5f}  ({strftime(time_elapsed)})')

def test_model(mdl: Any, device: torch.device) -> None:
    '''Test the given model on the ADNI test dataset.'''
    test_loader = create_test_dataloader()
    mdl.eval()
    time_start = time.time()
    with torch.no_grad():
        total = 0
        correct = 0
        for images, labels in tqdm(test_loader):
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
    mdl = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
    mdl.to(device)
    train_model(mdl, epochs=args.epochs, device=device)
    test_model(mdl, device=device)
    save_model(mdl)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('epochs', type=int)
    args = parser.parse_args()
    main(args)
