# training, validating, testing and saving the model
from dataset import *
from modules import *
import torch
from torch import optim

def train(model: TripletNetwork, criterion: TripletLoss, optimiser: optim.Optimizer,
          train_loader: DataLoader, valid_loader: DataLoader, epochs: int):
    losses = {
        "train": [],
        "valid": []
    }
    train_set_size = len(train_loader) # no of batches
    valid_set_size = len(valid_loader)
    print(f"Training images: {train_set_size*BATCH_SIZE}")
    print(f"Number of training batches: {train_set_size}")
    for epoch in range(epochs):
        # training
        epoch_train_loss = 0
        model.train()
        for batch_no, (a_t, label, p_t, n_t) in enumerate(train_loader):
            # move the data to the GPU

            # zero the gradients
            optimiser.zero_grad()
            # input triplet images into model
            a_out_t, p_out_t, n_out_t = model(a_t, p_t, n_t)
            # calculate the loss
            loss_t = criterion(a_out_t, p_out_t, n_out_t)
            # backpropagate
            loss_t.backward()
            # step the optimiser
            optimiser.step()
            # add the loss
            epoch_train_loss += loss_t.item()

            print(f"Batch {batch_no + 1}, Loss: {loss_t.item()}")
            if batch_no > 3:
                break 

        # record average training loss over epoch
        losses["train"].append(epoch_train_loss/train_set_size)

        # validation
        epoch_valid_loss = 0
        model.eval()
        for batch_no, (a_v, label, p_v, n_v) in enumerate(valid_loader):
            # move the data to the GPU

            # input triplet images into model
            a_out_v, p_out_v, n_out_v = model(a_v, p_v, n_v)
            # calculate the loss
            loss_v = criterion(a_out_v, p_out_v, n_out_v)
            # add the loss
            epoch_valid_loss += loss_v.item()

            print(f"Batch {batch_no + 1}, Loss: {loss_v.item()}")
            if batch_no > 3:
                break 

        # record average training loss over epoch
        losses["valid"].append(epoch_valid_loss/valid_set_size)

    return losses


if __name__ == '__main__':
    # setup the transforms for the images
    transform = transforms.Compose([
        transforms.Resize((256, 240)),
        transforms.ToTensor(),
        OneChannel()
    ])
    # set up the datasets
    train_set = TripletDataset(root="data/train", transform=transform)
    valid_set = TripletDataset(root="data/valid", transform=transform)
    test_set = TripletDataset(root="data/test", transform=transform)

    # set up the dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    # set up network and hyperparameters
    model = TripletNetwork()
    criterion = TripletLoss()
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 1

    losses = train(model, criterion, optimiser, train_loader, valid_loader, epochs)
    print(losses)