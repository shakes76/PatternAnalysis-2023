#Imports
import torch
import torch.optim as optim
import torch.nn as nn
from modules import Perceiver
from dataset import ADNI
import matplotlib.pyplot as plt

#Hyperparameter
LATENT_DIMENTIONS = 128 # Dimensions in higher dimensionality space
EMBEDDED_DIMENTIONS = 32 # Dimensions in lower dimensionality space
CROSS_ATTENTION_HEADS = 1 # How many cross attentions to use in model
SELF_ATTENTION_HEADS = 4 # How many self-attentions to run per level
SELF_ATTENTION_DEPTH = 4 # How many times to run the self-attention
MODEL_DEPTH = 4 # The number of times to repeat the block of cross-attention, self-attention and latent transformer
EPOCHS = 40 # Amount of epochs to run the training for
BATCH_SIZE = 5 #Batch size of images used
LEARNING_RATE = 0.0004 # Learning rate for training
SAVE_PATH_LOCATION = "./MODEL.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using ", device)

#Initialise the perceiver to hyper parameters
perceiver = Perceiver(MODEL_DEPTH, SELF_ATTENTION_DEPTH, LATENT_DIMENTIONS, EMBEDDED_DIMENTIONS, CROSS_ATTENTION_HEADS, SELF_ATTENTION_HEADS)
perceiver.to(device)

#Dataset
adni = ADNI(BATCH_SIZE) # Initialises the ADNI dataset class
trainloader = adni.training_data_loader # Pulls data from the adni data set for training
loss_function = nn.CrossEntropyLoss() # Using cross entropy loss function
eps = 1e-8
optimiser = optim.Adam(perceiver.parameters(), lr=LEARNING_RATE, eps=eps) # Using adam optimiser as most general optimiser

# Training results will be stored in these lists
loss_data = [] # stores data related to the loss during each epoch
accuracy = [] # stores the accuracy of the model during training


for epoch in range(EPOCHS):
    correct = 0
    total = 0
    running_loss = 0
    for data in trainloader:
        inputs, labels = data
        inputs = inputs.to(device) # sends the input images for first batch to gpu
        labels = labels.to(device) # sends labels to gpu for batch
        optimiser.zero_grad()
        outputs = perceiver(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimiser.step()
        predicted = torch.max(outputs.data, 1)[1]
        total += labels.shape[0]
        correct += torch.sum(predicted == labels).item()
        running_loss += loss.item()

    print("Epoch " + str(epoch) + " finished")
    loss_data.append(running_loss / total)
    accuracy.append(correct / total)

print("The loss data is ", loss_data)
print("The accuracy data is ", accuracy)

plt.plot(loss_data)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over epochs')
plt.show()  

plt.plot(accuracy)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training accuracy over epochs')
plt.show()

plt.plot(loss_data, label="Loss data", color="blue")
plt.plot(accuracy, label="Accuracy data", color="orange")
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.title("Accuracy vs Loss")
plt.legend()
plt.show()

torch.save(perceiver.state_dict(), SAVE_PATH_LOCATION)