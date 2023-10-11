import torch
from torch import nn
from torch.utils.data import DataLoader
from Siamese.dataset import get_test_dataset
from modules import SiameseNetwork


def test(model, device, test_loader):
    # Load the model
    model.eval()
    test_loss = 0
    correct = 0

    criterion = nn.BCELoss()

    with torch.no_grad():
        for (images_1, images_2, targets) in test_loader:
            images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
            outputs = model(images_1, images_2).squeeze()
            test_loss += criterion(outputs, targets).sum().item()  # sum up batch loss
            pred = torch.where(outputs > 0.5, 1, 0)  # get the index of the max log-probability
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32

    model = SiameseNetwork()
    model_path = "siamese_network.pt"
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    train_data = get_test_dataset('E:/comp3710/AD_NC')
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)

    test(model, device, train_dataloader)

