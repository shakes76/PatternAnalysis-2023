import torchvision
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import numpy as np


from modules import SiameseModel, RawSiameseModel, ContrastiveLossFunction
from dataset import load_train_data, load_test_data


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

            break
        break

    end = time.time() #time generation

    print("Finish training")
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total")


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