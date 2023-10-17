#!/home/terym/miniconda3/envs/pytorch/bin/python3

import torch
from modules import ImpUNet, DiceLoss
from dataset import ISICDataset
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision.utils import save_image

import sys

import matplotlib.pyplot as plt

BATCH_SIZE = 5
LEARNING_RATE = 0.005
NUM_EPOCH = 20 
SPLIT_RATIO = 0.8

device = ('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((512, 512))])

train_data = ISICDataset(img_dir="data/train_data", truth_dir="data/train_truth", transform=transform, train=True)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

total_step = len(train_loader)

lambda_lr = lambda epoch: LEARNING_RATE * (0.985**epoch)

model = ImpUNet(3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) 
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
loss_fcn = DiceLoss()

model.train()
for epoch in range(NUM_EPOCH):
    print(f"epoc : {epoch} of {NUM_EPOCH}")
    for i, images in enumerate(train_loader):
        img = images[0].to(device)
        truth = images[1].to(device)

        # forward pass
        outputs = model(img)
        loss = loss_fcn(outputs, truth)
        
        #backwards and optimize
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step() 

        # print status
        if (i + 1) % 10 == 0:
            print("Epoch: [{}/{}], Step: [{}/{}], Loss: {:.5f}"
                  .format(epoch+1, NUM_EPOCH, i+1, total_step, loss.item()))
            sys.stdout.flush()
            outputs = outputs.round()
            saved = torch.cat((outputs, truth), dim=0)
            save_image(saved.view(-1, 1, 512, 512), f"data/prod_img/{epoch}_{i+1}_seg.png", nrow=5)

        # scheduler step
        scheduler.step()

torch.save(model.state_dict(), "model_dict.pt")
