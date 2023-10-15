import os

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from modules import SiameseNetwork
from dataset import get_train_dataset


def train(model, dataloader, device, optimizer, epoch):

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

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(images_1), len(dataloader.dataset),
                       100. * batch_idx / len(dataloader), loss.item()))


def validate(model, dataloader, device):
    print("Starting Validation.")
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in dataloader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(dataloader.dataset)

    print('\nVal set: Average loss: {:.4f}, Validation Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(dataloader.dataset),
        100. * correct / len(dataloader.dataset)))

    print("Finished Validation.")


if __name__ == '__main__':
    epochs = 10
    batch_size = 256
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model, criterion, and optimizer
    model = SiameseNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Get dataloaders
    print("Loading data...")
    train_data = get_train_dataset('E:/comp3710/AD_NC')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    print("Data loaded.")

    print("Starting training.")
    for epoch in range(1, epochs + 1):
        train(model, train_dataloader, device, optimizer, epoch)
    print("Finished training.")

    save_directory = "E:/PatternAnalysis-2023/results"
    save_filename = f"siamese_network_{epochs}epochs(2).pt"

    # 创建保存目录，如果它不存在
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 完整的保存路径
    save_path = os.path.join(save_directory, save_filename)

    torch.save(model.state_dict(), save_path)
