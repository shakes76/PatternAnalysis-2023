import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from modules import SiameseNetwork
from dataset import get_train_dataset


def train(model, dataloader, device, optimizer, epoch, batch_size):

    model.train()

    #  using `BCELoss` as the loss function
    criterion = nn.BCELoss()

    for batch_idx, (images_1, images_2, targets) in enumerate(dataloader):
        images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images_1, images_2).squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))


def validate(model, dataloader, criterion):
    # TODO: Implement validation logic for one epoch
    pass


def main(epochs=10, batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, criterion, and optimizer
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters())

    # Get dataloaders
    train_data = get_train_dataset('E:/comp3710/AD_NC')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        train(model, train_dataloader, device, optimizer, epoch, batch_size)

    torch.save(model.state_dict(), "siamese_network.pt")


if __name__ == '__main__':
    main()
