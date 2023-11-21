"""
Author: Rohan Kollambalath
Student Number: 46963765
COMP3710 Sem2, 2023.
Script to train the perceiver transformer model. 
"""

import dataset as ds
from modules import *
import torch.optim as optim
from predict import test_accuracy, visualize_loss, visualize_accuracies, valid_accuracy

# check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(epochs, learning_rate, batch_size, model):
    """
    Method that trains a model with the ADNI dataset for expected batch size. The method gets the train 
    and validation datasets and uses them to train model for number of epochs. BCELoss and Adam optimiser 
    are used. After each epoch a validation accuracy is calculated and the batch loss is added to a list.
    The model returns the loss and accuracies. The function also saves the model preiodically when it 
    reaches new highs in accuracy.
    """
    # retrieve dataset
    dataset = ds.ADNI_Dataset(); train_loader, validation_loader = dataset.get_train_and_valid_loader()
    # BCELoss used with the Adam optimiser
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.BCELoss()
    
    # lists to store the results of training
    batch_losses = []
    accuracies = []
    
    max_accuracy = 0

    # main training loop
    for epoch in range(epochs):
        batch_loss = 0
        # iterate through the loeader in train mode
        model.train()
        for j, (images, labels) in  enumerate(train_loader):
            # model cannot take batches smaller than specified batch size
            if images.size(0) == batch_size:
                # forward pass through model
                optimizer.zero_grad()
                images = images.to(device); labels = labels.to(device)
                outputs = model(images)
                # calculate loss and backpropogate
                loss = criterion(outputs.to(torch.float), labels.to(torch.float))
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                        
                
        # Print results for the epoch        
        batch_losses.append(batch_loss/(j+1))
        print("epoch {} complete".format(epoch + 1))
        print("loss is {}".format(batch_loss/(j+1)))
        # Get accuracy for the epoch
        accuracy = valid_accuracy(model, batch_size, validation_loader)
        print("validation accuracy is {}".format(accuracy))
        accuracies.append(accuracy) 
        
        # save the model if validation accuracy is higher that before
        if accuracy > max_accuracy:
            torch.save(model.state_dict(), "model/model.pth")
            max_accuracy = accuracy


        
    return model, batch_losses, accuracies


if __name__ == "__main__":
    """
    Main driver that trains the models for the specified hyperparameters. Model is trained and then 
    saved before calculating the accuracy on the test set and visualising loss and validation accuracies
    """
    # hyper parameters
    epochs = 10
    depth = 3
    learning_rate = 0.0005
    batch_size = 32
    # model parameters
    LATENT_DIM = 32
    LATENT_EMB = 64
    latent_layers = 4
    latent_heads = 8
    classifier_out = 16
    batch_size = 32
    
    # instantiate the model and move it to GPU
    model = ADNI_Transformer(depth, LATENT_DIM, LATENT_EMB, latent_layers, latent_heads, classifier_out, batch_size)
    model.to(device=device)
    
    # train and save the model
    model, losses, train_accuracies = train(epochs, learning_rate, batch_size, model)
    torch.save(model.state_dict(), "model/model.pth")
    
    # visualise the results 
    print("testing final model accuracy")
    accuracy = test_accuracy(model, batch_size)
    print("Accuracy of model is {}%".format(accuracy))
    visualize_loss(losses)
    visualize_accuracies(train_accuracies)


    
