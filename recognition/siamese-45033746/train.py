from modules import SiameseNetwork, BinaryClassifier
from torch.utils.data import DataLoader
from torch import optim
from torch.nn import TripletMarginLoss
import torch
from utils import save_plot
from dataset import load
from os.path import exists
import torch.nn as nn

"""
train.py

load ADNI data sets from dataloader.py and utilise to train siamese cnn and binary classifier nn.
"""

SIAMESE_MODEL_PATH = "assets/siamese_model.pth"
BINARY_MODEL_PATH = "./assets/binary_model.pth"
EPOCHS = 1


def iterate_batch(title: str, dataLoader: DataLoader, criterion: TripletMarginLoss, opt, counter: [],
                  loss: [], epoch: int, device, model: SiameseNetwork):
    """
    Iterate over a dataloaders batches with a model
    :param title: label to discriminate print to console as training or validation sets
    :param dataLoader: DataLoader containing batches of images to use on model
    :param criterion: TripletMarginLoss contrastive loss function
    :param opt: Adam optimiser
    :param counter: Tracks number of total batches run
    :param loss: Tracks loss per batch
    :param epoch: Discriminates epoch number when printed to console
    :param device: gpu device
    :param model: SiameseNetwork to train or validate
    :return: counter [] containing total batches run thus far, loss [] tracking contrastive loss per batch
    """
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

        # print(f"Epoch {epoch} - Siamese {title} Batch {i} : Loss = {loss_contrastive.item()}\n")
        counter.append(i)
        loss.append(loss_contrastive.item())

    return counter, loss


def train_siamese(model: SiameseNetwork, criterion: TripletMarginLoss, trainDataLoader: DataLoader,
                  validDataLoader: DataLoader, epochs: int, device):
    """
    Train the SiameseNet from modules.py with cross-validation and save model to ./assets after
    :param model: Instance of SiameseNet sent to GPU
    :param criterion: TripletMarginLoss to measure contrastive loss
    :param trainDataLoader: DataLoader containing the batches for the training set
    :param validDataLoader: DataLoader containing the batches for the validation set
    :param epochs: Number of epochs to train for
    :param device: GPU
    """
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

        print(f"Epoch {epoch}, average siamese training loss = {sum(loss) / len(loss)}")

        # Iterate over cross validation batch
        model.eval()
        counter, loss = iterate_batch("Validation", validDataLoader, criterion, optimiser, val_counter,
                                      val_loss, epoch, device, model)
        val_counter = val_counter + counter
        val_loss = val_loss + loss

        print(f"Epoch {epoch}, average siamese validation loss = {sum(loss) / len(loss)}")

    save_plot(train_counter, train_loss, "siamese_train")
    save_plot(val_counter, val_loss, "siamese_validation")

    torch.save(model.state_dict(), SIAMESE_MODEL_PATH)


def train_binary(model: BinaryClassifier, siamese: SiameseNetwork, criterion: nn.BCELoss,
                 trainDataLoader: DataLoader, validDataLoader: DataLoader, epochs: int, device):
    """
    Train BinaryClassifier from modules.py to take the SiameseNet's out features and determine Alzheimer's class AD or
    NC. Saves model to ./assets after
    :param model: Instance of BinaryClassifier sent to GPU
    :param siamese: Instance of a trained SiameseNet
    :param criterion: BCEWithLoss
    :param trainDataLoader: DataLoader containing the batches for the training set
    :param validDataLoader: DataLoader containing the batches for the validation set
    :param epochs: Number of epochs to train for
    :param device: GPU
    :return:
    """
    train_counter = []
    val_counter = []
    train_loss = []
    val_loss = []

    optimiser = optim.Adam(model.parameters(), lr=0.0005)

    siamese.eval()

    for epoch in range(epochs):
        loss = []

        # Iterate over training batches
        model.train()
        for i, (label, anchor, _, _) in enumerate(trainDataLoader, 0):
            # Send data to GPU
            anchor, label = anchor.to(device), torch.unsqueeze(label.to(device), dim=1).float() # label.to(device)

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

            # print(f"Epoch {epoch} - Binary Training Batch {i} : Loss = {loss.item()}\n")
            train_counter.append(i)
            train_loss.append(loss.item())

            loss.append(loss.item())

        print(f"Epoch {epoch}, average binary training loss = {sum(loss) / len(loss)}")

        model.eval()
        for i, (label, anchor, _, _) in enumerate(validDataLoader, 0):
            # Send data to GPU
            anchor, label = anchor.to(device), torch.unsqueeze(label.to(device), dim=1).float()

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

            # print(f"Epoch {epoch} - Binary Validation Batch {i} : Loss = {loss.item()}\n")
            val_counter.append(i)
            val_loss.append(loss.item())
            loss.append(loss.item())
        print(f"Epoch {epoch}, average binary validation loss = {sum(loss) / len(loss)}")

    save_plot(train_counter, train_loss, "binary_train")
    save_plot(val_counter, val_loss, "binary_validation")

    torch.save(model.state_dict(), BINARY_MODEL_PATH)


def parent_train_binary(device, train: DataLoader, val: DataLoader):
    """
    Helper function for training the BinaryClassifier, ensures that a pre-trained model of SiameseNet exists, or
    creates a new one.
    :param device: GPU
    :param train: DataLoader containing the batches for the training set
    :param val: DataLoader containing the batches for the validation set
    :return:
    """
    # Send classifier to gpu
    net = BinaryClassifier().to(device)

    siamese_net = None

    # Check siamese model has been trained
    siamese_exists = exists(SIAMESE_MODEL_PATH)
    if not siamese_exists:
        print("No SiameseNet trained model found, training new SiameseNet")
        # Send model to gpu
        net = SiameseNetwork().to(device)
        train_siamese(net, TripletMarginLoss(), train, val, EPOCHS, device)
    else:
        print("Trained SiameseNet found, loading now...")

    siamese_net = SiameseNetwork()
    siamese_net.load_state_dict(torch.load(SIAMESE_MODEL_PATH))
    siamese_net.to(device)

    crit = nn.BCELoss()
    train_binary(net, siamese_net, crit, train, val, EPOCHS, device)


def main():
    """
    Main function to load in ADNI data, configure gpu or cpu devices, and train SiameseNet and BianryClassifier with.
    :return:
    """
    trainData, valData = load()
    print(f"Data loaded")

    # Device configuration
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train binary classifier based on siamese classifier
    parent_train_binary(gpu, trainData, valData)


if __name__ == '__main__':
    main()
