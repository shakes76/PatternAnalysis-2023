from glob import glob
import torch
import torch.nn as nn
import dataset
import modules
import utils
import predict

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


TRAINDATA = "ISIC/ISIC-2017_Training_Data/ISIC-2017_Training_Data"
TESTDATA = "ISIC/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data"
VALIDDATA = "ISIC/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data"
TRAINTRUTH = "ISIC/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth"
TESTTRUTH = "ISIC/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth"
VALIDTRUTH = "ISIC/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth"

CUDA_DEVICE_NUM = 0
NUM_EPOCHS = 15
BATCH_SIZE = 1
WORKERS = 4
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    train_dataset = dataset.CustomDataset(image_dir = TRAINDATA,
                                mask_dir=TRAINTRUTH,
                                transform=transforms.Compose([transforms.ToTensor()]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True)
    model = modules.UNet(3,1,[64,128,256,512]) 
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, 
                                momentum = 0.9, weight_decay = 5e-4)
    total_steps = len(train_dataloader)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr = LEARNING_RATE, 
    #                                steps_per_epoch = total_steps, epochs = NUM_EPOCHS)
    model.train()
    for epoch in range(NUM_EPOCHS):
        print("EPOCH:",epoch)
        for i, batch in enumerate(train_dataloader):
            images = batch['image']
            masks = batch['mask']
            #print(images.shape)
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)
            #print(images.shape)
            outputs = model(images)
            
            #print(1,outputs)
            #print(2,masks)
            loss = criterion(outputs, masks)
            #print(i)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{total_steps}], Loss: {loss.item():.5f}')
            #scheduler.step()
            utils.save_predictions_as_imgs(train_dataloader,model)

    FILE = "model.pth"
    torch.save(model.state_dict(), FILE)

    loaded_model = modules.UNet(3,1,[64,128,256,512]) 
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()


if __name__ == "__main__":
    main()