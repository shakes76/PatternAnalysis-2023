import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os

import CONSTANTS
from dataset import load_test_data
from modules import SiameseTwin, SimpleMLP, SiameseNeuralNet
from utils import load_from_checkpoint

def load_backbone(filename:str):
    """
    loads a trained overall Siamese network and extracts the backbone
    args:
        filename: the name of the checkpoint file only. Do not include the path.
            the filepath should be defined in CONSTANTS.py
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Loading classifier to device: ", device)

    siamese = SiameseNeuralNet()
    siamese = siamese.to(device)
    optimiser = torch.optim.Adam(siamese.parameters(), lr=0.0001)

    start_epoch, siamese, optimiser, training_losses, eval_losses = load_from_checkpoint(filename, siamese, optimiser)
    backbone = siamese.get_backbone()

    print(f"Trained backbone loaded successfully from {CONSTANTS.MODEL_PATH + filename}")
    print(backbone)

    return backbone, device

def load_trained_classifier(filename:str):
    """
    loads a trained classifier network
    args:
        filename: the name of the checkpoint file only. Do not include the path.
            the filepath should be defined in CONSTANTS.py
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Loading classifier to device: ", device)
    
    classifier = SimpleMLP()
    classifier = classifier.to(device)
    optimiser = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start_epoch, classifier, optimiser, training_losses, eval_losses = load_from_checkpoint(filename, classifier, optimiser)

    print(f"Trained classifier loaded successfully from {CONSTANTS.MODEL_PATH + filename}")
    print(classifier)

    return classifier, device

def make_predictions(classifier, backbone, device, random_seed=None):
    """
    applies the trained models to predict the entire test set and provides a text-based summary of the results
    this operation is deterministic if a random seed is provided
    args:
        classifier: the trained classifier network
        backbone: the trained embedding network
        device: the device to run the inferencing on
        random_seed: the random seed to use for reproducibility
    """

    # For reproducibility if desired. RNG seeding is handled in load_test_data()
    if random_seed is not None:
        torch.use_deterministic_algorithms(True)

    classifier.eval()
    backbone.eval()

    # Load test data for inferencing
    test_loader = load_test_data(Siamese=False, random_seed=random_seed)
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
    """
    applies the trained models to predict a sample of the test set and provides a visualisation of the results
    this operation is deterministic if a random seed is provided
    args:
        classifier: the trained classifier network
        backbone: the trained embedding network
        device: the device to run the inferencing on
        random_seed: the random seed to use for reproducibility
        save_name: the filename to save the visualisation as. if None, the visualisation will be shown instead of saved
            do not include the path. the filepath should be defined in CONSTANTS.py. 
    """
    # For reproducibility if desired. RNG seeding is handled in load_test_data()
    if random_seed is not None:
        torch.use_deterministic_algorithms(True)

    classifier.eval()
    backbone.eval()

    # Load test data for inferencing
    test_loader = load_test_data(Siamese=False, random_seed=random_seed)
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
    figure = plt.figure(figsize=(15, 12))
    cols, rows = 6, 4

    labels_map = {0: 'AD', 1: 'NC'}

    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        plt.title(f"Actual label: {labels_map[labels[i].cpu().tolist()]}, \n Prediction: {labels_map[predicted[i].cpu().tolist()]}")
        plt.axis("off")
        plt.imshow(np.transpose(images[i].cpu().squeeze(), (1,2,0)), cmap="gray")
    
    if save_name is None:
        print("Showing visualisations for sample predictions")
        plt.show()
    else:
        plt.savefig(CONSTANTS.RESULTS_PATH + save_name, dpi=300)
        print(f"Visualisations for sample predictions saved to {CONSTANTS.RESULTS_PATH + save_name}")

if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    backbone, device = load_backbone("SiameseNeuralNet_checkpoint.tar")
    classifier, device = load_trained_classifier("SimpleMLP_checkpoint.tar")
    make_predictions(classifier, backbone, device, random_seed=64)
    visualise_sample_predictions(classifier, backbone, device, random_seed=64, save_name='Predictions')