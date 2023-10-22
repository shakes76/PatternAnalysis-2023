from modules import SiameseNetwork
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import TripletMarginLoss
import torch
from utils import show_plot, save_plot
from dataset import load


# 256x240

def iterate_batch(title: str, dataLoader: DataLoader, criterion: TripletMarginLoss, opt, counter: [],
                  loss: [], epoch: int, device, model : SiameseNetwork):
    # Iterate over batch
    for i, (label, anchor, positive, negative) in enumerate(dataLoader, 0):
        # Send data to GPU
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # Zero gradients
        opt.zero_grad()

        # Pass in anchor, positive, and negative into network
        anchor_vec, positive_vec, negative_vec = model(anchor, positive, negative)

        # Pass vectors and label to the loss function
        loss_contrastive = criterion(anchor_vec, positive_vec, negative_vec)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        opt.step()

        # Every batche print out the loss

        print(f"Epoch {epoch} - {title} Batch {i} : Loss = {loss_contrastive.item()}\n")
        counter.append(i)
        loss.append(loss_contrastive.item())

    return counter, loss


def train(model: SiameseNetwork, criterion: TripletMarginLoss, optimiser,
          trainDataLoader: DataLoader, validDataLoader: DataLoader, epochs: int, device):
    train_counter = []
    val_counter = []
    train_loss = []
    val_loss = []

    print(f"Training images : {len(trainDataLoader)*trainDataLoader.batch_size}")
    print(f"Validation images : {len(validDataLoader)*validDataLoader.batch_size}")

    for epoch in range(epochs):
        # Iterate over training batch
        model.train()
        counter, loss = iterate_batch("Training", trainDataLoader, criterion, optimiser, train_counter,
                                                  train_loss, epoch, device, model)
        train_counter = train_counter + counter
        train_loss = train_loss + loss

        # Iterate over cross validation batch
        model.eval()
        counter, loss = iterate_batch("Validation", validDataLoader, criterion, optimiser, val_counter,
                                              val_loss, epoch, device, model)
        val_counter = val_counter + counter
        val_loss = val_loss + loss

    save_plot(train_counter, train_loss, "train")
    save_plot(train_counter, train_loss, "validation")


if __name__ == '__main__':
    trainData, valData = load()
    print(f"Data loaded")
    # Device configuration
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Send model to cpu
    net = SiameseNetwork().to(gpu)
    # net = SiameseNetwork().cuda()

    adam_optimiser = optim.Adam(net.parameters(), lr=0.0005)
    epochs = 1

    train(net, TripletMarginLoss(), adam_optimiser, trainData, valData, epochs, gpu)
