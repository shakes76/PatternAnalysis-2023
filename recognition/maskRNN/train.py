from dataset import isicData
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
from module import loadModel
import torch
import torchvision
from torchvision.ops import box_iou
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = isicData("./dataset/ISIC-2017_Training_Data", "./dataset/ISIC-2017_Training_Part1_GroundTruth", "./dataset/ISIC-2017_Training_Part3_GroundTruth.csv",)
test_data = isicData("./dataset/ISIC-2017_Test_v2_Data", "./dataset/ISIC-2017_Test_v2_Part1_GroundTruth","./dataset/ISIC-2017_Test_v2_Part3_GroundTruth.csv",)

train_data = torch.utils.data.Subset(train_data, range(50))
train_loader = DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
test_data = torch.utils.data.Subset(test_data, range(50))
test_loader = DataLoader(test_data, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

model = loadModel()
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.0025, weight_decay=0.0001, momentum=0.9)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    count = 0
    for images, targets in train_loader:
        print(count, len(train_loader))
        count+=1
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    lr_scheduler.step()



model.eval()  # Set the model to evaluation mode
image, targets = test_data[3]
predictions = model([image])
print(predictions)
boxes = predictions[0]["boxes"]
scores = predictions[0]["scores"]
iou = box_iou(boxes, targets["boxes"])
idx = torch.argmax(iou)
print("IoU:", iou[idx])