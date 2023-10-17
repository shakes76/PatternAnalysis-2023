#Imports
import torch
import torch.optim as optim
import torch.nn as nn
from modules import Perceiver
from dataset import ADNI
import matplotlib.pyplot as plt

#Hyperparameter
#Currently example paramaters need to find good values
#Whole file needs changing
LATENT_DIMENTIONS = 128 # Dimensions in higher dimensionality space
EMBEDDED_DIMENTIONS = 32 # Dimensions in lower dimensionality space
CROSS_ATTENTION_HEADS = 1
SELF_ATTENTION_HEADS = 4
SELF_ATTENTION_DEPTH = 4
MODEL_DEPTH = 2
EPOCHS = 300
BATCH_SIZE = 5
LEARNING_RATE = 0.0004
SAVE_PATH_LOCATION = "./MODEL.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ", device)
perceiver = Perceiver(MODEL_DEPTH, SELF_ATTENTION_DEPTH, LATENT_DIMENTIONS, EMBEDDED_DIMENTIONS, CROSS_ATTENTION_HEADS, SELF_ATTENTION_HEADS)
perceiver.to(device)
# model_depth, self_attention_depth, latent_dimensions, embedded_dimensions, cross_attention_heads, self_attention_heads):

#Dataset
adni = ADNI(BATCH_SIZE)
trainloader = adni.training_data_loader
loss_function = nn.CrossEntropyLoss() #USE BCE
#loss_function1 = nn.BCELoss()
eps = 1e-8
optimiser = optim.Adam(perceiver.parameters(), lr=LEARNING_RATE, eps=eps)


loss_data = []
accuracy = []


for epoch in range(EPOCHS):
    correct = 0
    total = 0
    running_loss = 0
    for data in trainloader:
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = [a.to(device) for a in data]
        # inputs, labels = data
        # inputs = inputs.to(device)
        # labels = labels.to(device)

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward + backward + optimize
        outputs = perceiver(inputs)

        loss = loss_function(outputs, labels) #try this with the bce loss function, f.cross_entrpy()
        loss.backward()
        optimiser.step()

        _, predicted = torch.max(outputs.data, 1)
        #predicted = torch.max(outputs.data, 1)[1]
        total += labels.size(0)
        #total += labels.shape[0]
        correct += (predicted == labels).sum().item()
        #correct += torch.sum(predicted == labels).item()

        running_loss += loss.item()

    print(f"Epoch {epoch} completed")
    loss_data.append(running_loss / 500)
    accuracy.append(correct / total)

print("loss data ", loss_data)
print("accuracy data ", accuracy)

plt.plot(loss_data)
plt.xlabel('EPOCH')
plt.ylabel('Average Loss')
plt.show()   

plt.plot(accuracy)
plt.xlabel('EPOCH')
plt.ylabel('Training Accuracy')
plt.show()

torch.save(perceiver.state_dict(), SAVE_PATH_LOCATION)