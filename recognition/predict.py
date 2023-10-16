import torch
import torch.nn as nn
from dataset import ADNI
from modules import Perceiver

MODEL = "./MODEL.txt"


#Hyperparameter
#Currently example paramaters need to find good values
LATENT_DIMENTIONS = 128 # Dimensions in higher dimensionality space
EMBEDDED_DIMENTIONS = 32 # Dimensions in lower dimensionality space
CROSS_ATTENTION_HEADS = 1
SELF_ATTENTION_HEADS = 4
SELF_ATTENTION_DEPTH = 4
MODEL_DEPTH = 4
BATCH_SIZE = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ", device)
perceiver = Perceiver(MODEL_DEPTH, SELF_ATTENTION_DEPTH, LATENT_DIMENTIONS, EMBEDDED_DIMENTIONS, CROSS_ATTENTION_HEADS, SELF_ATTENTION_HEADS)
perceiver.load_state_dict(torch.load(MODEL))

correct_predictions = 0
total_predictions = 0

test_dataset = ADNI(BATCH_SIZE).testing_data_loader
with torch.no_grad():
    for index, value in enumerate(test_dataset):
        (images, labels) = value
        results = perceiver(images)
        #change below
        _, predicted = torch.max(results.data, 1)
        total_predictions +=labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        if (index == 50):
            break

print("Total correct detections are", correct_predictions)
print("Total predictions are", total_predictions)
print("Accuracy is", correct_predictions/total_predictions)

perceiver.to(device)

