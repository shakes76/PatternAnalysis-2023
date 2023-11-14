
'''
Test driver script for Mask R-CNN with ISIC 2017 dataset
  Outputs: model file 'Mask_s4580286_Final.pt' 
        boxplot of IoU
        training and testing loss curve  

'''
from dataset import MoleData
from modules import load_model
from train import train_model
from predict import evaluate_single_sample, display_single_sample
import torch
from torchvision.ops import nms, box_iou
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import time


def main():
    #Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print("Started")

    #Load Model
    model = load_model()
    model = model.to(device)

    train_data = MoleData("/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Training_Data",
                        "/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Training_Part1_GroundTruth",
                        "/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Training_Part3_GroundTruth.csv")
    train_data = torch.utils.data.Subset(train_data, range(500))
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x:tuple(zip(*x)),
        )

    test_data =MoleData("/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Test_v2_Data",
        "/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Test_v2_Part1_GroundTruth",
        "/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Test_v2_Part3_GroundTruth.csv",
        )
    test_data = torch.utils.data.Subset(test_data, range(100))
    test_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x:tuple(zip(*x)),
        )

    training_loss, testing_loss= train_model(model, train_dataloader, test_dataloader,device)
    #Save training and testing loss for later
    df = pd.DataFrame()
    df['testing_loss'] = testing_loss
    df["training_loss"] = training_loss
    df.to_csv("TestTrainFinal.csv")

    torch.save(model.state_dict(), "Mask_s4580286_Final.pt")

    #Predict
    model = load_model()
    model.float()
    model.load_state_dict(torch.load("Mask_s4580286_Final.pt"))
    model.eval()

    #Load dataset
    val_data =MoleData("/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Validation_Data",
    "/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Validation_Part1_GroundTruth",
    "/content/drive/MyDrive/ColabNotebooks/ISIC-2017-DATA/ISIC-2017_Validation_Part3_GroundTruth (1).csv",
    )
    val_data = torch.utils.data.Subset(val_data, range(100))
    val_dataloader = torch.utils.data.DataLoader(
      val_data,
      batch_size=1,
      shuffle=True,
      collate_fn=lambda x:tuple(zip(*x)),
      )

    accuracy = []
    iou = []
    for index, data in enumerate(val_data):
      image, target = data
      predictions = model([image])
      temp_correct, temp_iou = evaluate_single_sample(image,target, predictions)
      print(temp_correct)
      if temp_correct == 1 and index < 10:
        plt.figure(figsize=(5,10))
        display_single_sample(image,target, predictions, index)
      accuracy.append(temp_correct)
      iou.append(temp_iou)

    # Boxplot the IOU
    plt.figure()
    plt.boxplot(iou,sym='')
    plt.xticks(ticks=[1],labels=["Test Data"])
    plt.ylabel("IOU")
    plt.title("Boxplot of IOU")
    plt.grid("both")
    plt.show()

    # Label accuracy calculations
    print("ACCURACY", sum(accuracy)/len(accuracy))
    return 0


if __name__ == "__main__":
  main()