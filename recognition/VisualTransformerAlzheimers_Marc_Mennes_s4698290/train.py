import dataset
import modules
import torchvision
from torch.utils.data import DataLoader
import torch
import time

CROPSIZE = 240
PATHTODATASET = "/home/marc/Documents/PatternAnalysisReport/PatternAnalysis-2023/recognition/VisualTransformerAlzheimers_Marc_Mennes_s4698290/ADNI_AD_NC_2D"
#PATHTODATASET = "/home/Student/s4698290/report/ADNI"
ENCODERDENSELAYERS = [[800, 1000, 800]]*3
LR = 0.000001
BATCHSIZE = 128
EPOCHS = 40

#use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)
print('cuda' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.CenterCrop(CROPSIZE),
     torchvision.transforms.Lambda(lambda x: x/255), #use the format of image data between 0 and 1 not 0 and 255
     torchvision.transforms.Normalize(0.1236, 0.2309), #normalize the image (values determined from function in dataset.py)
     torchvision.transforms.Lambda(lambda x: x.unfold(1,CROPSIZE//10, CROPSIZE//10).unfold(2,CROPSIZE//10, CROPSIZE//10)),#split the image into 9 patches
     torchvision.transforms.Lambda(lambda x: x[0])#removes the color channel dimension as this is greyscale
    ]
)

trainData = dataset.ADNI(PATHTODATASET, transform=transform)
validationData = dataset.ADNI(PATHTODATASET, transform=transform, validation=True)
trainLoader = DataLoader(trainData, batch_size=BATCHSIZE, shuffle=True)
validationLoader = DataLoader(validationData, batch_size=BATCHSIZE, shuffle=True)

transformer = modules.ADNIConvTransformer(0, [800, 1000, 800], ENCODERDENSELAYERS).to(device)

loss = torch.nn.BCELoss()
optimizer = torch.optim.Adam(transformer.parameters(), lr=LR)

transformer.train()

trainAccuracies = []
valAccuracies = []
valLoss = []
trainLoss = []
startTime = time.time()
print("training...")
#train the model
for epoch in range(EPOCHS):
    
    batchAccuracies = []
    l = 0
    for batch in trainLoader:

        images, labels = batch[0].to(device), batch[1].to(device)

        outputs = transformer(images)
        l = loss(outputs[:, 0], labels.float())

        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        #gather training statistics
        predictions = torch.round(outputs[:, 0].detach()).to(torch.uint8)
        trainAccuracy = torch.sum(predictions == labels)/(labels.size()[0])
        batchAccuracies.append(trainAccuracy)
        trainLoss.append(l.item())
        
    trainAccuracies.append(sum(batchAccuracies)/len(batchAccuracies))

    torch.save(transformer, "transformer_model")

    torch.no_grad()
    transformer.eval()
    batchAccuracies = []
    #run through validation set
    for batch in validationLoader:
        images, labels = batch[0].to(device), batch[1].to(device)

        outputs  = transformer(images)
        l = loss(outputs[:, 0], labels.float())

        predictions = torch.round(outputs[:, 0]).to(torch.uint8)
        
        valAccuracy = torch.sum(predictions == labels)/(labels.size()[0])
        batchAccuracies.append(valAccuracy)
        valLoss.append(l.item())
    
    valAccuracies.append(sum(batchAccuracies)/len(batchAccuracies))
        

    torch.enable_grad()
    transformer.train()

    print("Epoch: {}/{}, final batch loss: {}, average accuracy: {}, validation accuracy: {}".format(epoch+1, EPOCHS, l.item(), trainAccuracies[-1], valAccuracies[-1]))

torch.save(torch.FloatTensor(trainAccuracies), "./trainaccdata")
torch.save(torch.FloatTensor(valAccuracies), "./valaccdata")
torch.save(torch.FloatTensor(trainLoss), "./trainlossdata")
torch.save(torch.FloatTensor(valLoss), "./vallossdata")
print("done.")
endTime = time.time()
print("took", (endTime-startTime), "seconds")


torch.save(transformer, "transformer_model")
"""
torch.no_grad()
transformer.eval()

print("testing...")
testAccuracies = []
for batch in testLoader:

    images, labels = batch[0].to(device), batch[1].to(device)

    predictions = torch.round(transformer(images)[:, 0]).to(torch.uint8)
    testAccuracy = torch.sum(predictions == labels)/(labels.size()[0])
    testAccuracies.append(testAccuracy)
    

print(sum(testAccuracies)/len(testAccuracies))"""
