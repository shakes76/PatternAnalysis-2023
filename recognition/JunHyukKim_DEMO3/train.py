from glob import glob
import torch
import torch.nn as nn
import dataset
import modules

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


TRAINDATA = "ISIC/ISIC-2017_Training_Data/ISIC-2017_Training_Data"
TESTDATA = "ISIC/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data"
VALIDDATA = "ISIC/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data"
TRAINTRUTH = "ISIC/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth"
TESTTRUTH = "ISIC/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth"
VALIDTRUTH = "ISIC/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth"


NUM_EPOCHS = 15
BATCH_SIZE = 4
WORKERS = 4
LEARNING_RATE = 0.0001
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#m = nn.Dropout(p=0.2)

class ContextLayer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(ContextLayer,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
    )
    def forward(self, x):
        return self.conv(x)
    
class ConvLayer(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(ConvLayer,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=2,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class Localization(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Localization,self).__init__()
        self.loc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, padding=1,stride=1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
    )
    def forward(self, x):
        return self.loc(x)
    
class Upsampling(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Upsampling,self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, padding=1,stride=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, input_channels=3,out_channels=1,features=[64,128,256,512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2,2)
        self.first_conv = nn.Conv2d(input_channels, 64, 3, padding=1,stride=1),
        self.first_context = ContextLayer(64, 64)
        self.second_context = ContextLayer(128, 128)
        self.third_context = ContextLayer(256, 256)
        self.fourth_context = ContextLayer(512, 512)
        
        self.first_down = nn.ConvTranspose2d(64, 128, kernel_size=3,stride=2)
        self.second_down = nn.ConvTranspose2d(128,256,kernel_size=3,stride=2)
        self.third_down = nn.ConvTranspose2d(256,512,kernel_size=3,stride=2)
        self.fourth_down = nn.ConvTranspose2d(512,512*2,kernel_size=3,stride=2)

        for feature in features:
            self.downs.append(ContextLayer(input_channels, feature))    
            input_channels = feature    
        
        self.first_up = Upsampling(512*2,512)
        self.second_up = Upsampling(512,256)
        self.thrid_up = Upsampling(256,128)
        self.fourth_up = Upsampling(128,64)

        self.first_local = Localization(512,512)
        self.second_local = Localization(256,256)
        self.third_local = Localization(128,128)
        
        for feature in reversed(features):
            print()
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2,feature,kernel_size=2,stride=2
                )
            )
            self.ups.append(ContextLayer(feature*2,feature))
        self.bottleneck = ContextLayer(features[-1],features[-1]*2)
        self.final_conv = nn.Conv2d(features[0],out_channels,kernel_size=1)
        self.final_activiation = nn.Sigmoid()
        #self.l1 = nn.Linear(input_size, hidden_size) 
    
    
    def forward(self, x):
        x = self.first_conv(x)
        context1 = self.first_context(x)
        x.add(context1)
        skip_connections1 = x

        x = self.second_down(x)
        context2 = self.second_context(x)
        x.add(context2)        
        skip_connections2 = x

        x = self.third_down(x)
        context3 = self.thrid_context(x)
        x.add(context3)        
        skip_connections3 = x

        x = self.third_down(x)
        context4 = self.fourth_context(x)
        x.add(context4)        
        skip_connections4 = x

        

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
        
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)

            x = self.ups[idx+1](concat_skip)
    

        return self.final_activiation(self.final_conv(x))



def main():
    train_dataset = dataset.CustomDataset(image_dir = TRAINDATA,
                                mask_dir=TRAINTRUTH,
                                transform=transforms.Compose([transforms.ToTensor()]))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                            shuffle=True)
    model = UNet(3,1,[64,128,256,512]) 
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
            modules.save_predictions_as_imgs(train_dataloader,model)

    FILE = "model.pth"
    torch.save(model.state_dict(), FILE)

    loaded_model = UNet(3,1,[64,128,256,512]) 
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()


if __name__ == "__main__":
    main()