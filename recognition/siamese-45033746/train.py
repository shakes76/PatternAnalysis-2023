from modules import SiameseNetwork, BinaryClassifier
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import TripletMarginLoss
import torch
from utils import save_plot
from dataset import load
from os.path import exists

MODEL_PATH = "./assets/siamese_model.pth"


def iterate_batch(title: str, dataLoader: DataLoader, criterion: TripletMarginLoss, opt, counter: [],
                  loss: [], epoch: int, device, model: SiameseNetwork):
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

        # Every batch print out the loss

        print(f"Epoch {epoch} - Siamese {title} Batch {i} : Loss = {loss_contrastive.item()}\n")
        counter.append(i)
        loss.append(loss_contrastive.item())

    return counter, loss


def train_siamese(model: SiameseNetwork, criterion: TripletMarginLoss, trainDataLoader: DataLoader,
                  validDataLoader: DataLoader, epochs: int, device):
    train_counter = []
    val_counter = []
    train_loss = []
    val_loss = []

    optimiser = optim.Adam(model.parameters(), lr=0.0005)

    print(f"Training images : {len(trainDataLoader) * trainDataLoader.batch_size}")
    print(f"Validation images : {len(validDataLoader) * validDataLoader.batch_size}")

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

    save_plot(train_counter, train_loss, "siamese_train")
    save_plot(val_counter, val_loss, "siamese_validation")

    torch.save(model.state_dict(), MODEL_PATH)

    return model


def train_binary(model: BinaryClassifier, siamese: SiameseNetwork, criterion: TripletMarginLoss,
                 trainDataLoader: DataLoader, validDataLoader: DataLoader, epochs: int, device):
    train_counter = []
    val_counter = []
    train_loss = []
    val_loss = []

    optimiser = optim.Adam(model.parameters(), lr=0.0005)

    siamese.eval()

    for epoch in range(epochs):

        # Iterate over training batches
        model.train()
        for i, (label, anchor, _, _) in enumerate(trainDataLoader, 0):
            # Send data to GPU
            anchor, label = anchor.to(device), label.to(device)

            # Zero gradients
            optimiser.zero_grad()

            # Generate siamese model embeddings from image data
            siamese_embeddings = siamese.forward_once(anchor)

            # Train binary classifier with siamese embeddings
            vec = model(siamese_embeddings)

            # Calculate loss
            loss = criterion(vec, label)

            # Calculate the backpropagation
            loss.backward()

            # Optimise
            optimiser.step()

            print(f"Epoch {epoch} - Binary Training Batch {i} : Loss = {loss.item()}\n")
            train_counter.append(i)
            train_loss.append(loss.item())

        model.eval()
        for i, (label, anchor, _, _) in enumerate(validDataLoader, 0):
            # Send data to GPU
            anchor, label = anchor.to(device), label.to(device)

            # Zero gradients
            optimiser.zero_grad()

            # Generate siamese model embeddings from image data
            siamese_embeddings = siamese.forward_once(anchor)

            # Train binary classifier with siamese embeddings
            vec = model(siamese_embeddings)

            # Calculate loss
            loss = criterion(vec, label)

            # Calculate the backpropagation
            loss.backward()

            # Optimise
            optimiser.step()

            print(f"Epoch {epoch} - Binary Validation Batch {i} : Loss = {loss.item()}\n")
            val_counter.append(i)
            val_loss.append(loss.item())

    save_plot(train_counter, train_loss, "binary_train")
    save_plot(val_counter, val_loss, "binary_validation")


def parent_train_siamese(device, train: DataLoader, val: DataLoader):
    # Send model to gpu
    net = SiameseNetwork().to(device)

    return train_siamese(net, TripletMarginLoss(), train, val, 15, device)


# model: BinaryClassifier, siamese: SiameseNetwork, criterion: TripletMarginLoss,
#                  trainDataLoader: DataLoader, validDataLoader: DataLoader, epochs: int, device
def parent_train_binary(device, train: DataLoader, val: DataLoader):
    # Send classifier to gpu
    net = BinaryClassifier().to(device)

    siamese_net = None

    # Check siamese model has been trained
    siamese_exists = exists(MODEL_PATH)
    if not siamese_exists:
        print("No SiameseNet trained model found, training new SiameseNet")
        siamese_net = parent_train_siamese(device, train, val)
    else:
        print("Trained SiameseNet found, loading now...")
        siamese_net = SiameseNetwork()
        siamese_net.load_state_dict(torch.load(MODEL_PATH))
        siamese_net.to(device)

    train_binary(net, siamese_net, TripletMarginLoss(), train, val, 15, device)


def main():
    trainData, valData = load()
    print(f"Data loaded")

    # Device configuration
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train binary classifier based on siamese classifier
    parent_train_binary(gpu, trainData, valData)
    #parent_train_siamese(gpu, trainData, valData)


if __name__ == '__main__':
    main()
