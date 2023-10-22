import matplotlib.pyplot as plt
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
perceiver.to(device)
correct_predictions = 0
total_predictions = 0
running_accuracy = []

test_dataset = ADNI(BATCH_SIZE).testing_data_loader
with torch.no_grad():
    for index, value in enumerate(test_dataset):
        (images, labels) = value
        images = images.to(device)
        labels = labels.to(device)
        results = perceiver(images)
        #change below
        #_, predicted = torch.max(results.data, 1)
        predicted = torch.max(results.data, 1)[1]
        predicted = predicted.to(device)
        #total_predictions +=labels.size(0)
        total_predictions += labels.shape[0]
        #correct_predictions += (predicted == labels).sum().item()
        correct_predictions += torch.sum(predicted == labels).item()
        running_accuracy.append(correct_predictions / total_predictions)

print("Total correct detections are", correct_predictions)
print("Total predictions are", total_predictions)
print("Accuracy is", correct_predictions/total_predictions)


plt.plot(correct_predictions, label="Correct predictions", color="blue")
plt.plot(total_predictions, label="Total predictions", color="orange")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title("Accuracy")
plt.legend()
plt.show()

plt.plot(running_accuracy)
plt.xlabel('Epoch')
plt.ylabel('Correct predictions')
plt.title("Accuracy")
plt.show()

