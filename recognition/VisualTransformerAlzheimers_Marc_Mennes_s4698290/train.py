"""
Training code for the visual transformer
All hyperparameters are set to those used to achieve the results in README.md
"""
import dataset
import modules
import torchvision
from torch.utils.data import DataLoader
import torch
import time

CROPSIZE = 240
PATHTODATASET = "ADNI_AD_NC_2D" #replace with wherever your data set is
ENCODERDENSELAYERS = [[512, 512, 512]]*3
LR = 0.00001
BATCHSIZE = 128
EPOCHS = 60

#use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)
print('cuda' if torch.cuda.is_available() else 'cpu')

#dataset mean: 31.5226 dataset std: 58.8811

#transformations for the training set
trainTransform = torchvision.transforms.Compose(
    [torchvision.transforms.CenterCrop(CROPSIZE),
     torchvision.transforms.RandAugment(num_ops=4), #augment the data with random transforms
     torchvision.transforms.Lambda(lambda x: x/255), #bring image data between 1 and 0
     torchvision.transforms.Lambda(lambda x: x.unfold(1,CROPSIZE//4, CROPSIZE//4).unfold(2,CROPSIZE//4, CROPSIZE//4)),#split the image into 4 patches
     torchvision.transforms.Lambda(lambda x: x[0])#removes the color channel dimension as this is greyscale
    ]
)

#transformations for the validation set
valTransform = torchvision.transforms.Compose(
    [torchvision.transforms.CenterCrop(CROPSIZE),
     torchvision.transforms.Lambda(lambda x: x/255), #bring image data between 1 and 0
     torchvision.transforms.Lambda(lambda x: x.unfold(1,CROPSIZE//4, CROPSIZE//4).unfold(2,CROPSIZE//4, CROPSIZE//4)),#split the image into 4 patches
     torchvision.transforms.Lambda(lambda x: x[0])#removes the color channel dimension as this is greyscale
    ]
)

#load the datasets and torch data loaders
trainData = dataset.ADNI(PATHTODATASET, transform=trainTransform)
validationData = dataset.ADNI(PATHTODATASET, transform=valTransform, validation=True)
trainLoader = DataLoader(trainData, batch_size=BATCHSIZE, shuffle=True)
validationLoader = DataLoader(validationData, batch_size=BATCHSIZE, shuffle=True)

transformer = modules.ADNITransformer(16, 60, 4, 0.2, [512], ENCODERDENSELAYERS).to(device)

#other options for models:

#convolution transformer: 
#transformer = modules.ADNIConvTransformer(0.2, [512], ENCODERDENSELAYERS).to(device)

#experimental, got good validation accuracy but even less carry over to the test set than
#the standard ViT. If you want to use it you must comment out:

#torchvision.transforms.Lambda(lambda x: x.unfold(1,CROPSIZE//4, CROPSIZE//4).unfold(2,CROPSIZE//4, CROPSIZE//4)),
#torchvision.transforms.Lambda(lambda x: x[0])

#in both transforms above

#For continuing training on a previously trained model:
#transformer = torch.load("transformer_model")

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
    batchLoss = []
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
        batchLoss.append(l.item())
    
    trainLoss.append(sum(batchLoss)/len(batchLoss)) 
    trainAccuracies.append(sum(batchAccuracies)/len(batchAccuracies))

    torch.save(transformer, "transformer_model")

    torch.no_grad()
    transformer.eval()
    batchAccuracies = []
    batchLoss = []
    #run through validation set, gather statistics, then put model back in train mode
    for batch in validationLoader:
        images, labels = batch[0].to(device), batch[1].to(device)

        outputs  = transformer(images)
        l = loss(outputs[:, 0], labels.float())

        predictions = torch.round(outputs[:, 0]).to(torch.uint8)
        
        valAccuracy = torch.sum(predictions == labels)/(labels.size()[0])
        batchAccuracies.append(valAccuracy)
        batchLoss.append(l.item())
    
    valAccuracies.append(sum(batchAccuracies)/len(batchAccuracies))
    valLoss.append(sum(batchLoss)/len(batchLoss))
        

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
