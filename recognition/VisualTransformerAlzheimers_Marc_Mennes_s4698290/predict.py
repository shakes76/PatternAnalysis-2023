"""
Run prediction on the model trained and saved by train.py
"""
import dataset
import modules
import torchvision
from torch.utils.data import DataLoader
import torch

CROPSIZE = 240
PATHTODATASET = "ADNI_AD_NC_2D" #replace with wherever your data set is
BATCHSIZE = 128

#use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)
print('cuda' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.CenterCrop(CROPSIZE),
     torchvision.transforms.Lambda(lambda x: x/255), #use the format of image data between 0 and 1 not 0 and 255
     torchvision.transforms.Lambda(lambda x: x.unfold(1,CROPSIZE//4, CROPSIZE//4).unfold(2,CROPSIZE//4, CROPSIZE//4)),#split the image into 4x4 = 16 equal square patches
     torchvision.transforms.Lambda(lambda x: x[0])#removes the color channel dimension as this is greyscale
    ]
)

testData = dataset.ADNI(PATHTODATASET, transform=transform, test=True)
testLoader = DataLoader(testData, batch_size=BATCHSIZE, shuffle=True)

#load the model saved by train.py
transformer = torch.load("transformer_model").to(device)

torch.no_grad()
transformer.eval()

#run through the test set, record accuracies, print the average accuracy
print("testing...")
testAccuracies = []
for batch in testLoader:

    images, labels = batch[0].to(device), batch[1].to(device)

    predictions = torch.round(transformer(images)[:, 0]).to(torch.uint8)
    testAccuracy = torch.sum(predictions == labels)/(labels.size()[0])
    testAccuracies.append(testAccuracy)
    
print("final accuracy is:")
print(sum(testAccuracies)/len(testAccuracies))
