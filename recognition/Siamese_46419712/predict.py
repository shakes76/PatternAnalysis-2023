import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt
import random

from modules import RawSiameseModel, BinaryModelClassifier
from dataset import LoadData

SIAMESE_MODEL_SAVE_PATH = "siamese.pt"
CLASSIFIER_MODEL_SAVE_PATH = "classifier.pt"

def load_model(model_type=0):
    if model_type == 0:
        PATH = SIAMESE_MODEL_SAVE_PATH
    else:
        PATH = CLASSIFIER_MODEL_SAVE_PATH

    if os.path.exists(PATH):
        load_model = torch.load(PATH)
        model = load_model['model']
        return model
    else:
        return None

def test_model(model, cModel, test_loader):
    # evaluate the model
    model.eval()
    cModel.eval()
    with torch.no_grad(): # disable gradient computation
        random_batch = random.randint(0, len(test_loader) - 1)
        for batch_idx, (img, label) in enumerate(test_loader):
            if batch_idx == random_batch:
                img = img.to(device)
                label = label.to(device).float()

                fv = model(img)
                output = cModel(fv)
                output = output.view(-1)

                predicted = (output > 0.5).float()

                return img, label, predicted

def save_plot_image(img, label, predicted):

    pass
        
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    ######### LOADING DATA ##########
    print("Begin loading data")
    test_loader = LoadData(train=False).load_data()
    print("Finish loading data \n")

    siamese_model = RawSiameseModel().to(device)
    load_save_model = load_model()

    if load_save_model is not None:
        siamese_model.load_state_dict(load_save_model)
        print("load siamese model from save")
    else:
        print("error, unable to load, cannot find save file")


    classifier_model = BinaryModelClassifier().to(device)
    load_classifier_save_model = load_model(1)

    if load_classifier_save_model is not None:
        classifier_model.load_state_dict(load_classifier_save_model)
        print("Load classifier model from save")
    else:
        print("error, unable to load cannot find save file")

    img, label, predicted = test_model(siamese_model, classifier_model, test_loader)

    save_plot_image(img, label, predicted)
