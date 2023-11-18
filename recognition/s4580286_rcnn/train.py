
'''
Trains the model on the 2017 ISIC dataset
Outputs: Trained Model -> Mask_s4580286_Final.pt
        Training and testing loss -> TestTrainFinal.csv
'''

from dataset import MoleData
from modules import load_model
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from tqdm import tqdm


print(torch.__version__)

#Hyper Parameters
num_epochs = 30
learning_rate = 0.0025
num_classes = 3

def unpack_images(images, device):
  new_images = []
  for image in images:
    new_images.append(image.to(device))
  return new_images

def unpack_targets(targets, device):
  new_targets = []
  for target in targets:
    new_target = {}
    for k, v in target.items():
        new_target[k] = v.to(device)
    new_targets.append(new_target)
  return new_targets

def one_epoch(model, optimizer, dataloader, device, epoch, train_flag):
  total_losses = []
  i = 0
  total_step = len(dataloader)
  for i, (images,targets) in enumerate(dataloader): #load a batch
    images = unpack_images(images,device)
    targets = unpack_targets(targets, device)

    # Forward pass
    outputs = model(images, targets)
    # Backward and optimize
    if train_flag == 1:
      optimizer.zero_grad()
    total_loss = 0.0
    for loss in outputs.values():
      total_loss += loss

    if train_flag == 1:
      total_loss.backward()
      optimizer.step()

    total_losses.append(total_loss.detach().cpu().item())
    if (i+1) % 100 == 0:
        print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


  return total_losses


def train_model(model, train_dataloader, test_dataloader,device):
  optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,weight_decay=0.0001, momentum=0.9)
  #linear schedule - how to reduce learning rate as training
  total_step = len(train_dataloader)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=total_step, epochs=num_epochs)
  total_loss = []

  #Train
  model.train()
  training_loss = []
  testing_loss = []
  for epoch in range(num_epochs):
    print(">Training")
    start = time.time()
    train_loss = one_epoch(model, optimizer, train_dataloader, device, epoch, 1)
    scheduler.step()
    training_loss.append(sum(train_loss))
    print("TRAINING_LOSS", sum(train_loss))
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    #Testing
    print("> Testing")
    start = time.time() #time generation
    with torch.no_grad():
        all_losses = []
        test_loss = one_epoch(model, optimizer, train_dataloader, device, epoch,0)
        testing_loss.append(sum(test_loss))
        print("TESTING_LOSS", sum(test_loss))

    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
    print('END')
    fig, ax = plt.subplots()
    ax.plot(training_loss, label="Train")
    ax.plot(testing_loss, label="Test")
    ax.legend()

  return training_loss, testing_loss

def main():
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
  test_data = torch.utils.data.Subset(train_data, range(100))
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


if __name__ == "__main__":
  main()