import torch.nn as nn
import os
import numpy as np
import random
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
from torch.nn import BCELoss

from dataset import create_siamese_dataloader,get_classification_dataloader,get_transforms_training, get_transforms_testing
from modules import SiameseResNet, ContrastiveLoss, ClassifierNet

#--------- SET DEVICE --------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#-----------------------------------TRAINING THE SIAMESE NETWORK-----------------------

print("---------Training Siamese Network---------")

#-------- LOAD TRAINING DATA --------
ROOT_DIR_TRAIN = "/home/groups/comp3710/ADNI/AD_NC/train" # Modify Path if needed
train_loader, val_loader = create_siamese_dataloader(ROOT_DIR_TRAIN, batch_size=32, split_flag=True)

#---------- HYPERPARAMETERS --------
learning_rate =  0.1 
num_epochs = 40
margin = 1
print("learning rate for siamese:",learning_rate)

#--------- INITIATE SIAMESE MODEL --------------
model_siamese = SiameseResNet().to(device)

#---------- DEFINE LOSS AND OPTIMISER ------
optimizer = optim.Adam(model_siamese.parameters(), lr=learning_rate)
criterion = ContrastiveLoss(margin=margin)

#---------- TRAINING LOOP SIAMESE----------
print("---------Training Siamese Network---------")

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Set model to train mode
    model_siamese.train()
    running_loss = 0

    # Training 
    for batch_idx, (img1, img2, labels, _, _) in enumerate(train_loader):
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        optimizer.zero_grad()

        distances = model_siamese(img1, img2)

        # Backpropagate loss
        loss = criterion(distances, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Compute average training loss of epoch  
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss) 

    # Validation
    model_siamese.eval() # set model to test mode
    val_loss = 0.0
    with torch.no_grad():
         for batch_idx, (img1, img2, labels, _, _) in enumerate(val_loader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            distances = model_siamese(img1, img2)
            loss = criterion(distances, labels)
            
            val_loss += loss.item()

    # Compute average validation loss
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)  

    print('[Epoch %d] training loss: %.3f, validation loss: %.3f' % (epoch + 1, avg_train_loss, avg_val_loss))

print("Training complete Siamese")

# Save Model of the last epoch
print("Saved siamese after training")
siamese_path = 'siamese_40_3.pth'
torch.save(model_siamese.state_dict(), siamese_path)

# Plot losses
print("Saved loss plot Siamese")
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Training Loss", color="blue")
plt.plot(val_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot as an image
plt.savefig("siamese_loss_curves_siamese_40_3.png")





#--------------------------------TRAINING THE CLASSIFIER NETWORK-----------------------------------
print("---------Training Classifier Network---------")

#-------- LOAD TRAINING DATA --------
ROOT_DIR_TRAIN = "/home/groups/comp3710/ADNI/AD_NC/train"  # Modify Path if needed
train_loader, val_loader = get_classification_dataloader(ROOT_DIR_TRAIN, batch_size=32,split_flag=True)

#-------LOAD SIAMESE MODEL----------
siamese_model = SiameseResNet().to(device)
siamese_model.load_state_dict(torch.load(siamese_path, map_location=device))
siamese_model.eval() # set to eval mode just incase so weights are not updated

#--------- INITIATE CLASSIFIER MODEL --------------
# initiate classifier with the trained siamese model
classifier = ClassifierNet(siamese_model).to(device)

#---------- HYPERPARAMETERS ------------
learning_rate = 0.01
print("learning rate for classifier:",learning_rate)
num_epochs = 20


#---------- DEFINE LOSS AND OPTIMISER ------
criterion = BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=learning_rate,weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

#---------- TRAINING LOOP CLASSIFIER----------

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Initialize a variable to track the best validation accuracy
best_val_accuracy = 0.0
best_model_path = "best_classifier_model_40_20_2.pth"

for epoch in range(num_epochs):
    # Set classifier to train mode
    classifier.train() 
    running_loss = 0
    correct_train = 0
    total_train = 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.float().to(device)
        
        optimizer.zero_grad()

        outputs = classifier(imgs).squeeze()
        
        # Backpropagate loss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        preds = outputs.round()  # Round the predictions to get the class prediction 0/1
        correct_train += (preds == labels).float().sum() # compare predictions to the true label to calculate accuracy
        total_train += labels.size(0)

    # Compute average training loss of epoch and accurracy 
    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

    # Validation
    classifier.eval() # Set classfier to test mode
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
         for batch_idx, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.float().to(device)

            outputs = classifier(imgs).squeeze()
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()

            preds = outputs.round()  # Round the predictions to get the class prediction 0/1
            correct_val += (preds == labels).float().sum() # compare predictions to the true label to calculate accuracy
            total_val += labels.size(0)

    # Compute average validation loss and accurracy 
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct_val / total_val
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    # save the classifier with the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(classifier.state_dict(), best_model_path)
        

    scheduler.step()

    print(f'[Epoch {epoch + 1}] train loss: {avg_train_loss:.3f}, train accuracy: {train_accuracy:.3f}, val loss: {avg_val_loss:.3f}, val accuracy: {val_accuracy:.3f}')

print("Training complete")

# Save the trained classifier model after n epoch
classifier_save_path = "classifier_model_40_20_2.pth"
torch.save(classifier.state_dict(), classifier_save_path)
print(f"Saved classifier model to {classifier_save_path}")


# Plotting training and validation loss
plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Save the loss plot as an image
loss_plot_save_path = "classifier_loss_plot_classifier_model_40_20_2.png"
plt.savefig(loss_plot_save_path)
print(f"Saved loss plot to {loss_plot_save_path}")

plt.close()  

# Convert tensors to CPU and then to numpy arrays
train_accuracies_np = [acc.cpu().numpy() for acc in train_accuracies]
val_accuracies_np = [acc.cpu().numpy() for acc in val_accuracies]

# Plotting training and validation accuracy
plt.figure(figsize=(7, 5))
plt.plot(train_accuracies_np, label="Training Accuracy")
plt.plot(val_accuracies_np, label="Validation Accuracy")
plt.title("Accuracy vs. Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Save the accuracy plot as an image
accuracy_plot_save_path = "classifier_accuracy_plot_classifier_model_40_20_2.png"
plt.savefig(accuracy_plot_save_path)
print(f"Saved accuracy plot to {accuracy_plot_save_path}")

plt.close() 


#-------------------TESTING------------------------------

# Load test set
ROOT_DIR_TEST = "/home/groups/comp3710/ADNI/AD_NC/test"  # Modify Path if needed
test_loader = get_classification_dataloader(ROOT_DIR_TEST, batch_size=32,split_flag = False)

# Load the Classifier that has the best validation accuracy during training
classifier = ClassifierNet(siamese_model).to(device)
classifier.load_state_dict(torch.load(best_model_path, map_location=device))
classifier.eval()

test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        
        outputs = classifier(imgs).squeeze()
        
        # Compute loss
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        # Compute accuracy
        preds = outputs.round()
        correct_test += (preds == labels).float().sum()
        total_test += labels.size(0)

avg_test_loss = test_loss / len(test_loader)
test_accuracy = correct_test / total_test

print(f"Test Loss: {avg_test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")
