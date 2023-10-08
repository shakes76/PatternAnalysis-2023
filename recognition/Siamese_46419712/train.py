import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np
import os


from modules import SiameseModel, RawSiameseModel, ContrastiveLossFunction
from dataset import load_train_data, load_test_data

def save_model(siamese_epochs, siamese_model, siamese_criterion, siamese_optimizer,
                classifier_epochs=None, classifier_model=None, classifier_criterion=None, classifier_optimizer=None):
    if classifier_model is not None:
        torch.save({
            'siamese_epochs': siamese_epochs,
            'siamese_model': siamese_model.state_dict(),
            'siamese_criterion': siamese_criterion,
            'siamese_optimizer': siamese_optimizer.state_dict(),
            'classifier_epochs': classifier_epochs,
            'classifier_model': classifier_model.state_dict(),
            'classifier_criterion': classifier_criterion,
            'classifier_optimizer': classifier_optimizer.state_dict(),
        }, "model.pt")
    else:
        torch.save({
            'siamese_epochs': siamese_epochs,
            'siamese_model': siamese_model.state_dict(),
            'siamese_criterion': siamese_criterion,
            'siamese_optimizer': siamese_optimizer.state_dict()
        }, "model.pt")

    print("Save model")

def load_model():
    if os.path.exists("model.pt"):
        load_model = torch.load("model.pt")
        # ignore if classifier complete
        if 'classifier_model' in load_model:
            epoch1 = load_model['siamese_epochs']
            siamese = load_model['siamese_model']
            s_criterion = load_model['siamese_criterion']
            s_optimizer = load_model['siamese_optimizer']
            epoch2 = load_model['classifier_epochs']
            classifier = load_model['classifier_model']
            c_criterion = load_model['classifier_criterion']
            c_optimizer = load_model['classifier_optimizer']
            return epoch1, siamese, s_criterion, s_optimizer, epoch2, classifier, c_criterion, c_optimizer
        else:
            epoch1 = load_model['siamese_epochs']
            siamese = load_model['siamese_model']
            s_criterion = load_model['siamese_criterion']
            s_optimizer = load_model['siamese_optimizer']
            return epoch1, siamese, s_criterion, s_optimizer, None
    else:
        return None

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")

    ######### LOADING DATA ##########
    print("Begin loading data")
    train_loader = load_train_data()
    test_loader = load_test_data()

    print("Finish loading data")


    #########  TRAINING SIAMASE MODEL ##########
    # Testing model
    model = RawSiameseModel()

    # hyper parameters
    num_epochs = 20
    learning_rate = 0.1

    criterion = ContrastiveLossFunction()

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=total_step, epochs=num_epochs)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5) # Optimize model parameter

    #Piecewise Linear Schedule
    total_step = len(train_loader)
    # adjust the learning rate during training -> potentially help reach convergence state faster while also ensure it does not over train
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=total_step, epochs=num_epochs)
    

    loss_list = [] 
    step_count = 0
    start = time.time() #time generation

    model.train()
    print("Start training")
    for epoch in range(num_epochs):
        print(f"epoch: {epoch + 1}")
        
        for i, val in enumerate(train_loader):
            img0, img1 , labels = val
            img0 = img0.to(device)
            img1 = img1.to(device)
            labels = labels.to(device)

            output1 = model(img0)
            output2 = model(img1)
            loss = criterion(output1, output2, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()


            # save loss for graph
            loss_list.append(loss.item())
            step_count += 1

            if (i+1) % 100 == 0:
                print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                        .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


            scheduler.step()
            save_model(epoch, model, loss, optimizer)
            break
        break

    end = time.time() #time generation

    print("Finish training")
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total")

    #########  TRAINING BINARY CLASSIFIER MODEL ########## 
    print("Start classifier training")
    
    start = time.time()

    end = time.time() #time generation
    print("Finish classifier training")
    elapsed = end - start
    print("Training classifier took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total")

    #########  TESTING SIAMASE MODEL ##########
    print("Start Testing")
    
    start = time.time()
    # evaluate the model
    model.eval() # disable drop out, batch normalization
    with torch.no_grad(): # disable gradient computation
        correct_predict = 0
        total_test = 0
        # for img1, img2, labels in test_loader:
        #     img1 = img1.to(device)
        #     img2 = img2.to(device)
        #     labels = labels.to(device)
        #     outputs = model(img1, img2)
        #     _, predicted = torch.max(outputs.data, 1)
        #     total_test += labels.size(0)
        #     correct_predict += (predicted == labels).sum().item()
        
        # print('Test Accuracy: {} %'.format(100 * correct_predict / total_test))
    
    end = time.time() #time generation

    print("Finish Testing")

    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total")