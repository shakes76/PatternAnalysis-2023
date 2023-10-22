from dataset import get_test_set, get_patient_split, SiameseDataSet, compose_transform
from modules import SiameseNetwork
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import TripletMarginLoss
import torch
from utils import show_plot


# 256x240

def load():
    trainer, val = get_patient_split()
    transform = compose_transform()

    trainSet = SiameseDataSet(trainer, transform)
    valSet = SiameseDataSet(val, transform)
    testSet = SiameseDataSet(get_test_set(), transform)

    trainDataLoader = DataLoader(trainSet, shuffle=True, num_workers=2, batch_size=64)
    valDataLoader = DataLoader(valSet, shuffle=True, num_workers=2, batch_size=64)
    testDataLoader = DataLoader(testSet, shuffle=True, num_workers=2, batch_size=64)

    return trainDataLoader, valDataLoader, testDataLoader


def train(model: SiameseNetwork, criterion: TripletMarginLoss, optimiser,
          trainDataLoader: DataLoader, validDataLoader: DataLoader, epochs: int, device):
    counter = []
    loss_accumulator = []
    iteration = 0

    trainSize = len(trainDataLoader)
    validSize = len(validDataLoader)
    print(f"Training images : {trainSize * 64}")
    print(f"Number of training batches : {trainSize}")

    for epoch in range(epochs):

        # Iterate over batch
        for i, (label, anchor, positive, negative) in enumerate(trainDataLoader, 0):
            # Send data to GPU
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # Zero gradients
            optimiser.zero_grad()

            # Pass in anchor, positive, and negative into network
            anchor_vec, positive_vec, negative_vec = net(anchor, positive, negative)

            # Pass vectors and label to the loss function
            loss_contrastive = criterion(anchor_vec, positive_vec, negative_vec)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimiser.step()

            # Every 10 batches print out the loss
            if i % 10 == 0:
                print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
                iteration += 10

                counter.append(iteration)
                loss_accumulator.append(loss_contrastive.item())

    show_plot(counter, loss_accumulator)


if __name__ == '__main__':
    trainData, valData, testData = load()
    print(f"Data loaded")
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Send model to cpu
    net = SiameseNetwork().to(device)
    # net = SiameseNetwork().cuda()
    optimiser = optim.Adam(net.parameters(), lr=0.0005)
    epochs = 1

    train(net, TripletMarginLoss(), optimiser, trainData, valData, epochs, device)
