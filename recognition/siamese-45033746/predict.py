from modules import BinaryClassifier, SiameseNetwork
from torch.utils.data import DataLoader
from os.path import exists
from train import BINARY_MODEL_PATH, SIAMESE_MODEL_PATH
import torch
from dataset import get_test_set

"""
predict.py

load trained models and measure accuracy
"""

def load():
    # Device configuration
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not exists(BINARY_MODEL_PATH) or not exists(SIAMESE_MODEL_PATH):
        print("No trained models available, please run train.py to create trained models")

    bin_net = BinaryClassifier()
    bin_net.load_state_dict(torch.load(BINARY_MODEL_PATH))
    bin_net.to(gpu)

    siamese_net = SiameseNetwork()
    siamese_net.load_state_dict(torch.load(SIAMESE_MODEL_PATH))
    siamese_net.to(gpu)

    return bin_net, siamese_net, gpu


def predict(model: BinaryClassifier, net: SiameseNetwork, testDataLoader: DataLoader, device):
    model.eval()
    net.eval()

    for i, (label, anchor, _, _) in enumerate(testDataLoader, 0):
        anchor = anchor.to(device)

        siamese_embeddings = net.forward_once(anchor)

        result = model(siamese_embeddings)

        print(f"pred : {result}, label : {label}")


def main():
    test = get_test_set()
    bin, sin, device = load()
    predict(bin, sin, test, device)


if __name__ == "__main__":
    main()
