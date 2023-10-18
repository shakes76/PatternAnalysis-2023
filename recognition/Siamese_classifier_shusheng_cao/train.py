import torch
import os
import argparse
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms

from modules import SiameseNetwork, Classifier, TripletLoss
from dataset import ADNISiameseDataset, ADNIDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train Siamese Network
def train_siamese(data_dir, output_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1155,), (0.2254,))
    ])
    siamese_trainset = ADNISiameseDataset(data_dir, transform)
    trainloader = torch.utils.data.DataLoader(siamese_trainset, batch_size=32, shuffle=True, pin_memory=True)
    print("Data loaded")

    model = SiameseNetwork()
    model = model.to(device)

    scaler = GradScaler()

    criterion = TripletLoss()
    learning_rate = 0.1
    optimiser = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    epochs = 15

    total_step = epochs * len(trainloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=learning_rate,
                                                    max_momentum=0.9, total_steps=total_step)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for anchor, positive, negative in trainloader:
            anchor, positive, negative = anchor.to(device), positive.to(device), \
                negative.to(device)
            optimiser.zero_grad()

            with autocast():
                distance_positive = model(anchor, positive)
                distance_negative = model(anchor, negative)
                loss = criterion(distance_positive, distance_negative)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scheduler.step()
            scaler.update()

            running_loss += loss.item()
            print("loss: ", loss.item())

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    torch.save(model.state_dict(), output_path)

def train_classifier(data_dir, siamese_model, output_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1155,), (0.2254,))
    ])
    classifier_trainset = ADNIDataset(data_dir, transform)
    trainloader = torch.utils.data.DataLoader(classifier_trainset, batch_size=32, shuffle=True, pin_memory=True)
    print("Data loaded")

    model = Classifier(2)
    model.load_state_dict(siamese_model)
    model = model.to(device)

    scaler = GradScaler()

    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 0.1
    optimiser = torch.optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    epochs = 30

    total_step = epochs * len(trainloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=learning_rate,
                                                    max_momentum=0.9, total_steps=total_step)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimiser.zero_grad()

            with autocast():
                y_pred = model(inputs)
                loss = criterion(y_pred, labels)

            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scheduler.step()
            scaler.update()

            running_loss += loss.item()
            print("loss: ", loss.item())

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}")

    torch.save(model.state_dict(), output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="The file to train the model")

    parser.add_argument("network", choices=["siamese", "classifier"], help="Choose a network to train")
    parser.add_argument("--data_dir", default=os.path.join(".", "AD_NC", "train"), help="The data dir used to train")
    parser.add_argument("--output_path", default=os.path.join(".", "model.pth"))
    parser.add_argument("--siamese_model", default=os.path.join(".", "model.pth"))

    args = parser.parse_args()

    if args.network == "siamese":
        train_siamese(args.data_dir, args.output_path)
    elif args.network == "classifier":
        train_classifier(args.data_dir, args.siamese_model, args.output_path)