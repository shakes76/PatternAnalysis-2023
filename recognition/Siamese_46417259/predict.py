import torch
import matplotlib.pyplot as plt
import numpy as np
import time

import CONSTANTS
from dataset import load_data
from modules import SiameseTwin, SimpleMLP
from train import load_from_checkpoint

def load_backbone(filename:str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Loading classifier to device: ", device)

    backbone = SiameseTwin()
    backbone = backbone.to(device)
    optimiser = torch.optim.Adam(backbone.parameters(), lr=0.0001)

    start_epoch, backbone, optimiser, training_losses, eval_losses = load_from_checkpoint(filename, backbone, optimiser)

    print(f"Trained backbone loaded successfully from {CONSTANTS.MODEL_PATH + filename}")
    print(f"Backbone was trained for {start_epoch - 1} epochs")
    print(backbone)

    return backbone, device

def load_trained_classifier(filename:str):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Loading classifier to device: ", device)
    
    classifier = SimpleMLP()
    classifier = classifier.to(device)
    optimiser = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start_epoch, classifier, optimiser, training_losses, eval_losses = load_from_checkpoint(filename, classifier, optimiser)

    print(f"Trained classifier loaded successfully from {CONSTANTS.MODEL_PATH + filename}")
    print(f"Classifier was trained for {start_epoch - 1} epochs")
    print(classifier)

    return classifier, device

def make_predictions(classifier, backbone, device, random_seed=None):
    # For reproducibility if desired. RNG seeding is handled in load_data()
    if random_seed is not None:
        torch.use_deterministic_algorithms(True)

    classifier.eval()
    backbone.eval()

    # Load test data for inferencing
    test_loader = load_data(training=False, Siamese=False, random_seed=random_seed)
    print("Test data loaded successfully")

    start = time.time()

    print("Testing accuracy of classifier over all test images")
    with torch.no_grad(): # disables gradient calculation
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device).float()

            feature_vect = backbone(images)
            outputs = classifier(feature_vect)
            outputs = outputs.view(-1)

            predicted = (outputs > 0.5).float()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    end = time.time()
    elapsed = end - start

    print(f"{total} images were tested in {elapsed:.1f} seconds.")
    print(f"Number of correctly classified images: {correct}")
    print(f"Test Accuracy (rounded): {round(100 * correct / total, 4)} %")

def visualise_sample_predictions(classifier, backbone, device, random_seed=None, save_name=None):
    # For reproducibility if desired. RNG seeding is handled in load_data()
    if random_seed is not None:
        torch.use_deterministic_algorithms(True)

    classifier.eval()
    backbone.eval()

    # Load test data for inferencing
    test_loader = load_data(training=False, Siamese=False, random_seed=random_seed)
    print("Test data loaded successfully")

    with torch.no_grad(): # disables gradient calculation
        # get a batch of testing data, which is randomised by default
        images, labels = next(iter(test_loader))

        images = images.to(device)
        labels = labels.to(device).float()

        feature_vect = backbone(images)
        outputs = classifier(feature_vect)
        outputs = outputs.view(-1)

        predicted = (outputs > 0.5).float()
    
    # the following data visualisation code is modified based on code at
    # https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/data_tutorial.py
    # published under the BSD 3-Clause "New" or "Revised" License
    # full text of the license can be found in this project at BSD_new.txt
    figure = plt.figure(figsize=(12, 12))
    cols, rows = 4, 4

    labels_map = {0: 'AD', 1: 'NC'}

    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        plt.title(f"Actual label: {labels_map[labels[i].tolist()]}, prediction: {labels_map[predicted[i].tolist()]}")
        plt.axis("off")
        plt.imshow(np.transpose(images[i].squeeze(), (1,2,0)), cmap="gray")
    
    if save_name is None:
        print("Showing visualisations for sample predictions")
        plt.show()
    else:
        plt.savefig(CONSTANTS.RESULTS_PATH + save_name, dpi=300)
        print(f"Visualisations for sample predictions saved to {CONSTANTS.RESULTS_PATH + save_name}")

if __name__ == "__main__":
    backbone, device = load_backbone("SiameseNeuralNet_checkpoint.tar")
    classifier, device = load_trained_classifier("SimpleMLP_checkpoint.tar")
    make_predictions(classifier, backbone, device, random_seed=None)
    visualise_sample_predictions(classifier, backbone, device, random_seed=None, save_name=None)