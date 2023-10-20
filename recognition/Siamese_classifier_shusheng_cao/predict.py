import argparse
import os

import torch
import torchvision.transforms as transforms
from modules import SiameseNetwork, Classifier
from dataset import ADNIDataset, ADNISiameseDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_siamese(model, loader, threshold=0.6):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for anchor, positive, negative in loader:
            anchor, positive, negative = anchor.to(device), positive.to(device), \
                negative.to(device)
            distance_positive = model(anchor, positive)
            distance_negative = model(anchor, negative)
            similarity_positive = (distance_positive + 1) / 2
            similarity_negative = (distance_negative + 1) / 2
            correct += (similarity_positive > threshold).sum().item()
            correct += (similarity_negative < threshold).sum().item()
            total += 2 * anchor.size(0)
    return correct / total

def predict_siamese(data_dir, model_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1155,), (0.2254,))
    ])
    siamese_testset = ADNISiameseDataset(data_dir, transform)
    testloader = torch.utils.data.DataLoader(siamese_testset, batch_size=32, shuffle=True, pin_memory=True)
    print("Data loaded")

    model = SiameseNetwork()
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return evaluate_siamese(model, testloader)

def evaluate_classifier(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def predict_classifier(data_dir, model_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1155,), (0.2254,))
    ])
    siamese_testset = ADNIDataset(data_dir, transform)
    testloader = torch.utils.data.DataLoader(siamese_testset, batch_size=32, shuffle=True, pin_memory=True)

    model = Classifier(2)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    return evaluate_classifier(model, testloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The file to predict the model")

    parser.add_argument("network", choices=["siamese", "classifier"], help="Choose a network to predict")
    parser.add_argument("--data_dir", default=os.path.join(".", "AD_NC", "test"), help="The data dir used to predict")
    parser.add_argument("--model_path", default=os.path.join(".", "model.pth"))

    args = parser.parse_args()

    if args.network == "siamese":
        print("Accuracy: ", predict_siamese(args.data_dir, args.model_path))
    elif args.network == "classifier":
        print("Accuracy: ", predict_classifier(args.data_dir, args.model_path))
