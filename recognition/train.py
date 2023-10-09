#Imports
import torch
import torch.optim as optim
import torch.nn as nn
from modules import Perceiver
from dataset import ADNI
import matplotlib.pyplot as plt

#Hyperparameter
#Currently example paramaters need to find good values
LATENT_DIMENTIONS = 128 # Dimensions in higher dimensionality space
EMBEDDED_DIMENTIONS = 32 # Dimensions in lower dimensionality space
CROSS_ATTENTION_HEADS = 1
SELF_ATTENTION_HEADS = 4
SELF_ATTENTION_DEPTH = 4
MODEL_DEPTH = 4
EPOCHS = 2
BATCH_SIZE = 5
LEARNING_RATE = 0.004
SAVE_PATH = "./MODEL.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
perceiver = Perceiver(MODEL_DEPTH, SELF_ATTENTION_DEPTH, LATENT_DIMENTIONS, EMBEDDED_DIMENTIONS, CROSS_ATTENTION_HEADS, SELF_ATTENTION_HEADS)
perceiver.to(device)
# model_depth, self_attention_depth, latent_dimensions, embedded_dimensions, cross_attention_heads, self_attention_heads):

#Dataset
trainloader = ADNI(BATCH_SIZE)


for epoch in range(EPOCHS):
    correct = 0
    total = 0
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()
        optimiser.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()

    print(f"Epoch {epoch} completed")
    loss_data.append(running_loss / 500)
    accuracy.append(correct / total * 100)

plt.plot(loss_data)
plt.xlabel('EPOCH')
plt.ylabel('Average Loss')
plt.show()   

plt.plot(accuracy)
plt.xlabel('EPOCH')
plt.ylabel('Training Accuracy')
plt.show()  

torch.save(model.state_dict(), MODEL_PATH)