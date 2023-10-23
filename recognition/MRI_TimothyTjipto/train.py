'''Source code for training, validating, testing and saving your model. The model
should be imported from “modules.py” and the data loader should be imported from “dataset.py”. Make
sure to plot the losses and metrics during training'''

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import numpy as np
import matplotlib.pyplot as plt
import random
from datetime import datetime
from PIL import Image

import dataset 
from dataset import SiameseNetworkDataset1,SiameseNetworkDataset_test
import modules
from modules import SiameseNetwork, ContrastiveLoss
from predict import show_plot
import predict

# Paths of data 
TRAIN_PATH = "/home/Student/s4653241/AD_NC/train"
TEST_PATH = "/home/Student/s4653241/AD_NC/test"

INPUT_SHAPE= (120, 128) # SIZE OF IMAGE 256 X 240
BATCH_SIZE = 16 # Batch Size for DataLoader

TRAINING_MODE = False # Training mode
EPOCH_RANGE = 61 # Size of the Training Epoch
CHECKPOINT_TRAINING = False # Use Checkpoint and continue Training
LOAD_CHECKPOINT_TRAINING = "/home/Student/s4653241/MRI/Training_Epoch/Epoch_40.pth"
SAVE_EPOCH = True
EPOCH_SAVE__CHECKPOINT = 60  # Saves every 30 Epoch 

TEST_MODE = True # For Testing
CHECKPOINT = "/home/Student/s4653241/MRI/Training_Epoch/Epoch_60.pth" # Test the checkpoint you want
TEST_RANGE = 500 # Testing size
THRESHOLD = 0.5
VISUALISE = False # Print out Error pics DEBUGGING TOOL for now


def load_checkpoint(path):
    model = SiameseNetwork().cuda()
    optimizer = optim.Adam(model.parameters(), lr = 0.00006)

    device = torch.device("cuda")
    checkpoint = torch.load(LOAD_CHECKPOINT_TRAINING)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    counter = checkpoint['counter']
    loss = checkpoint['loss']
    iteration = checkpoint['iteration']
    model.train()
    return model,optimizer,epoch,counter,loss,iteration

def main():
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale with one channel
        transforms.Resize((120,128)),  # img_size should be a tuple like (128, 128) actual img(256x240)
        transforms.ToTensor(),
        # You can add more transformations if needed
    ])
    raw_dataset = datasets.ImageFolder(root=TRAIN_PATH)
    siamese_dataset = SiameseNetworkDataset1(raw_dataset, transform)
    # Create a simple dataloader just for simple visualization
    vis_dataloader = DataLoader(siamese_dataset,
                            shuffle=True,
                            num_workers=1,
                            batch_size=16)
    
    dataset.visualise_batch(vis_dataloader)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(), lr = 0.00006)
    current_epoch = 1

    counter = []
    loss_history = [] 
    iteration_number= 0

    if CHECKPOINT_TRAINING:

        net,optimizer,current_epoch,counter,loss_history,iteration_number=load_checkpoint(LOAD_CHECKPOINT_TRAINING)

    # Iterate throught the epochs
    if TRAINING_MODE:
        for epoch in range(current_epoch, EPOCH_RANGE):

            # Iterate over batches
            for i, (img0, img1, label) in enumerate(vis_dataloader, 0):

                # Send the images and labels to CUDA
                img0, img1, label = img0.cuda(), img1.cuda(), label.cuda()

                # Zero the gradients
                optimizer.zero_grad()

                # Pass in the two images into the network and obtain two outputs
                output1, output2 = net(img0, img1)

                # Pass the outputs of the networks and label into the loss function
                loss_contrastive = criterion(output1, output2, label)

                # Calculate the backpropagation
                loss_contrastive.backward()

                # Optimize
                optimizer.step()

                # Every 10 batches print out the loss
                if i % 50 == 0 :
                    counter.append(iteration_number)
                    loss_history.append(loss_contrastive.item())
                    iteration_number += 50

            if SAVE_EPOCH:
                if epoch%EPOCH_SAVE__CHECKPOINT == 0:
                    checkpoint  = {
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'counter':counter,
                    'loss': loss_history,
                    'iteration': iteration_number,
                    }
                    torch.save(checkpoint,  f"/home/Student/s4653241/MRI/Training_Epoch/Epoch_{epoch}.pth")
        
        print("End of Training")
        show_plot(counter, loss_history)
    if TEST_MODE:
        print(f'Start of testing {datetime.now()}')
        checkpoint = torch.load(CHECKPOINT)
        model = SiameseNetwork().cuda()
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        model.eval()

        raw_test_dataset = datasets.ImageFolder(root=TEST_PATH)

        test_siam = SiameseNetworkDataset_test(raw_test_dataset, transform)
        test_dataloader = DataLoader(test_siam,
                                shuffle=True,
                                num_workers=1,
                                batch_size=1)

        dataiter = iter(test_dataloader)
        x0, _, _,x0label,_ = next(dataiter)
    
        postive_prediction = 0
        Prediction = ['Different', 'Same']
        # Debugging 
        counter_same_but_wrong = 0
        counter_wrong_but_same = 0
        Neg_Neg = []
        Neg_Pos = []

        for i in range(TEST_RANGE):
            print(f'Iteration: {i}')
            PRINTING = True
            # Iterate over 10 images and test them with the first image (x0)
            _, x1, label2,_,x1label = next(dataiter)
            

            with torch.no_grad():   
                output1, output2 = model(x0.cuda(), x1.cuda())
            euclidean_distance = F.pairwise_distance(output1, output2)
           
            
            predict_class = predict.classify_pair(euclidean_distance.item(),THRESHOLD) # Threshold

            if predict_class == 1 and int(x0label) == int(x1label):
    
                postive_prediction+=1
                PRINTING = False
                
            if predict_class != 1 and int(x0label) != int(x1label):

                postive_prediction+=1
                PRINTING = False
            if PRINTING:
                print(f'Euclidean_distance: {euclidean_distance}') # Debugging Code
                print(f'x0 label: {int(x0label)}, x1 label: {int(x1label)}')
                print(f'Class predicted: {Prediction[predict_class]}')
                
                rounded_distance = round(float(euclidean_distance.item()), 1)

                if predict_class == 1 and int(x0label) != int(x1label):
                    counter_same_but_wrong += 1
                    if rounded_distance != 0.0:
                        Neg_Neg.append(rounded_distance)

                    
                
                if predict_class != 1 and int(x0label) == int(x1label):
                    counter_wrong_but_same += 1
                    
                    if rounded_distance != 1:
                        Neg_Pos.append(rounded_distance)

            if PRINTING and VISUALISE:
                plt.clf()
                plt.subplot(2, 8, 1)
                plt.title(int(x0label))
                x0_pic = transforms.ToPILImage()(x0[0])
                plt.axis('off')
                plt.imshow(x0_pic, cmap='gray')
                
                plt.subplot(2,8,2)
                plt.title(f'Dissimilarity: {euclidean_distance.item():.2f}\nClass predicted: {Prediction[predict_class]} ')
                plt.axis('off')
                


                plt.subplot(2, 8, 3)
                plt.title(int(x1label))
                x1_pic = transforms.ToPILImage()(x1[0])
                plt.axis('off')
                plt.imshow(x1_pic, cmap='gray')
               
                plt.savefig(f'/home/Student/s4653241/MRI/Test_pic/test{i}')
            
        
        Accuracy = postive_prediction/TEST_RANGE
        print(f'Using Checkpoint: {CHECKPOINT}\nAccuracy: {Accuracy}\nNo. of Positive Matches: {postive_prediction}\nNo. of Test: {TEST_RANGE}')
        


if __name__ == '__main__':
    main()