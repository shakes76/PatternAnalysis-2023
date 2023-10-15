import torch
import time
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import numpy as np

import random
from modules import RawSiameseModel, ContrastiveLossFunction, BinaryModelClassifier
from dataset import LoadData

SIAMESE_LOSS_SAVE_PATH = "siamese_loss_plot.png"
SIAMESE_MODEL_SAVE_PATH = "siamese.pt"

CLASSIFIER_LOSS_SAVE_PATH = "classifier_loss_plot.png"
CLASSIFIER_MODEL_SAVE_PATH = "classifier.pt"

def train_siamese(model, train_loader, criterion, optimizer, loss_list, scheduler):
    model.train()

    for i, val in enumerate(train_loader):
        img0, img1 , labels = val
        img0 = img0.to(device)
        img1 = img1.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output1 = model(img0)
        output2 = model(img1)
        loss = criterion(output1, output2, labels)

        loss.backward()

        optimizer.step()

        # save loss for graph
        loss_list.append(loss.item())
        
        # scheduler.step()


        if (i+1) % 40 == 0:
            print (">>>>> Step [{}/{}] Loss: {:.5f}"
                    .format(i+1, len(train_loader), loss.item()))

def validate_siamese(model, val_loader, criterion, val_loss_list):
    model.eval()

    with torch.no_grad():
        for i, val in enumerate(val_loader):
            img0, img1, label = val
            img0 = img0.to(device)
            img1 = img1.to(device)
            label = label.to(device)

            output1 = model(img0)
            output2 = model(img1)
            loss = criterion(output1, output2, label)
            val_loss_list.append(loss.item())

        if (i+1) % 10 == 0:
            print (">>>>> Step [{}/{}] Validate Loss: {:.5f}"
                    .format(i+1, len(val_loader), loss.item()))

def train_classifier(sModel, cModel, train_loader, criterion, optimizer, loss_list, scheduler):
    sModel.eval() # siamese
    cModel.train() # classifier

    for i, val in enumerate(train_loader):
        img, label = val
        img = img.to(device)
        label = label.to(device).float()
        optimizer.zero_grad()

        fv = sModel(img)
        output = cModel(fv)
        loss = criterion(output.view(-1), label)

        loss.backward()
        optimizer.step()

        # save loss for graph
        loss_list.append(loss.item())
        # scheduler.step()
        if (i+1) % 40 == 0:
            print (">>>>> Step [{}/{}] Loss: {:.5f}"
                    .format(i+1, len(train_loader), loss.item()))

def validate_classifier(sModel, cModel, val_loader, criterion, val_loss_list):
    sModel.eval()
    cModel.eval()

    with torch.no_grad():
        correct_predict = 0
        total_test = 0
        for i, (img, label) in enumerate(val_loader):
            img = img.to(device)
            label = label.to(device).float()

            fv = sModel(img) 
            output = cModel(fv)

            loss = criterion(output.view(-1), label)
            val_loss_list.append(loss.item())

            predicted = (output > 0.5).float()
            total_test += label.size(0)
            correct_predict += (predicted == label).sum().item()

            if (i+1) % 10 == 0:
                print (">>>>> Step [{}/{}] Classifier Validate Loss: {:.5f}"
                        .format(i+1, len(val_loader), loss.item()))

        print(f"Validate predict >>>> {correct_predict / total_test}%")

def test_model(model, cModel, test_loader):
    # evaluate the model
    model.eval() # disable drop out, batch normalization
    cModel.eval()
    with torch.no_grad(): # disable gradient computation
        correct_predict = 0
        total_test = 0
        for img, label in test_loader:
            img = img.to(device)
            label = label.to(device).float()

            fv = model(img)
            output = cModel(fv)
            output = output.view(-1)
            # print(output)
            predicted = (output > 0.5).float()
            print(">>>>> Predicted")
            print(predicted)

            print(">>>>> Actual")
            print(label)
            total_test += label.size(0)
            correct_predict += (predicted == label).sum().item()
    
    return correct_predict/total_test

def save_loss_plot(loss_list, epoch=0, train=True, siamese=True):
    """
        Save loss for siamese and classifier
    """
    plt.figure(figsize=(12, 8))
    plt.plot(loss_list, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    if train:
        if siamese:
            plt.title(f'Siamese Losses - Epoch {epoch + 1}')
            plt.savefig(SIAMESE_LOSS_SAVE_PATH)
        else:
            plt.title(f'Classifier Losses - Epoch {epoch + 1}')
            plt.savefig(CLASSIFIER_LOSS_SAVE_PATH)
    else:
        if siamese:
            plt.title('Siamese Losses Validate')
            plt.savefig('siamese_loss_plot_validate.png')
        else:
            plt.title('Classifier Losses Validate')
            plt.savefig('classifier_loss_plot_validate.png')
    plt.close()

def save_model(epochs, model, criterion, optimizer, scheduler, model_type=0):
    if model_type == 0:
        PATH = SIAMESE_MODEL_SAVE_PATH
    else:
        PATH = CLASSIFIER_MODEL_SAVE_PATH

    torch.save({
        "epochs": epochs,
        "model": model.state_dict(),
        "criterion": criterion,
        "optimizer": optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }, PATH)

    print("Save model")

# def load_model(model_type=0):
#     if model_type == 0:
#         PATH = SIAMESE_MODEL_SAVE_PATH
#     else:
#         PATH = CLASSIFIER_MODEL_SAVE_PATH

#     if os.path.exists(PATH):
#         load_model = torch.load(PATH)
#         epoch = load_model['epochs']
#         model = load_model['model']
#         criterion = load_model['criterion']
#         optimizer = load_model['optimizer']
#         scheduler = load_model['scheduler']

#         return epoch, model, criterion, optimizer, scheduler
#     else:
#         return None

def execute_sTrain(device, train_loader, val_loader):
    model = RawSiameseModel().to(device)

    # hyper-parameters
    num_epochs = 15
    learning_rate = 0.0001
    max_learning = 0.01

    criterion = ContrastiveLossFunction()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999)) # Optimize model parameter

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_learning, steps_per_epoch=len(train_loader), epochs=num_epochs) # temporary remove scheduler

    # training
    loss_list = []
    avg_loss_list = []

    # validating
    val_loss_list = []
    avg_val_loss = []

    start = time.time() #time generation

    print("Start training")

    for epoch in range(num_epochs):
        print ("Epoch [{}/{}]".format(epoch + 1, num_epochs))
        train_siamese(model, train_loader, criterion, optimizer, loss_list, scheduler)
        validate_siamese(model, val_loader, criterion, val_loss_list)

        save_model(epoch, model, criterion, optimizer, scheduler) # save model for every epoch

        avg_loss_list.append(np.mean(loss_list))
        avg_val_loss.append(np.mean(val_loss_list))

        loss_list = []
        val_loss_list = []
        save_loss_plot(avg_loss_list, epoch) # save loss plot for siamese train
        save_loss_plot(avg_val_loss, train=False) # save loss plot for siamese validate

    save_loss_plot(avg_loss_list, epoch) # save loss plot for siamese train
    save_loss_plot(avg_val_loss, train=False) # save loss plot for siamese validate

    end = time.time() #time generation

    print("Finish training")
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total \n")

    return model

def execute_cTrain(device, sModel, train_loader_classifier, val_loader_classifier):
    model = BinaryModelClassifier().to(device)

    # hyper parameters for classifier
    num_epochs = 50
    learning_rate = 0.0001
    max_learning = 0.01

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_learning, steps_per_epoch=len(train_loader_classifier), epochs=num_epochs) # temporary remove scheduler

    # training
    classifier_loss_list = []
    avg_classifier_loss_list = []

    # validating
    classifier_val_loss_list = []
    avg_classifier_val_loss = []

    print("Start classifier training")
    
    start = time.time()
    # train classifier
    for epoch in range(num_epochs):
        print ("Epoch [{}/{}]".format(epoch + 1, num_epochs))
        train_classifier(sModel, model, train_loader_classifier, criterion, optimizer, classifier_loss_list, scheduler)
        validate_classifier(sModel, model, val_loader_classifier, criterion, classifier_val_loss_list)
        save_model(epoch, model, criterion, optimizer, scheduler, 1) # save model for every epoch

        avg_classifier_loss_list.append(np.mean(classifier_loss_list))
        avg_classifier_val_loss.append(np.mean(classifier_val_loss_list))

        classifier_loss_list = []
        classifier_val_loss_list = []
        save_loss_plot(avg_classifier_loss_list, epoch, siamese=False) # save loss plot for classifier train
        save_loss_plot(avg_classifier_val_loss, epoch, train=False, siamese=False) # save loss plot for classifier val

    save_loss_plot(avg_classifier_loss_list, epoch, siamese=False) # save loss plot for classifier val
    save_loss_plot(avg_classifier_val_loss, epoch, train=False, siamese=False) # save loss plot for classifier val


    end = time.time() #time generation
    print("Finish classifier training")
    elapsed = end - start
    print("Training classifier took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total \n")

    return model

if __name__ == '__main__':
    random.seed(40)
    torch.manual_seed(40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    ######### LOADING DATA ##########
    print("Begin loading data")
    train_loader, val_loader = LoadData(train=True, siamese=True).load_data()
    train_loader_classifier, val_loader_classifier = LoadData(train=True, siamese=False).load_data()
    test_loader = LoadData(train=False).load_data()
    print("Finish loading data \n")

    #########  TRAINING SIAMASE MODEL ##########
    siamese_model = execute_sTrain(device, train_loader, val_loader)

    #########  TRAINING BINARY CLASSIFIER MODEL ########## 
    classifier_model = execute_cTrain(device, siamese_model, train_loader_classifier, val_loader_classifier)

    #########  TESTING MODEL ##########
    print("Start Testing")
    start = time.time()
    accuracy = test_model(siamese_model, classifier_model, test_loader)
    print('Test Accuracy: {} %'.format(100 * accuracy))
    end = time.time() #time generation
    print("Finish Testing")
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total")