import dataset
import torchvision
from torch.utils.data import DataLoader
import torch

CROPSIZE = 210
PATHTODATASET = "/home/marc/Documents/PatternAnalysisReport/PatternAnalysis-2023/recognition/VisualTransformerAlzheimers_Marc_Mennes_s4698290/ADNI_AD_NC_2D"

#use gpu if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.version.cuda)
print('cuda' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose(
    [torchvision.transforms.CenterCrop(CROPSIZE),
     torchvision.transforms.Lambda(lambda x: x/255), #use the format of image data between 0 and 1 not 0 and 255
     torchvision.transforms.Normalize(41.0344/255, 64.2557/255), #normalize the image (values determined from function in dataset.py)
     torchvision.transforms.Lambda(lambda x: x.unfold(1,CROPSIZE//3, CROPSIZE//3).unfold(2,CROPSIZE//3, CROPSIZE//3))#split the image into 9 patches
    ]
)

trainData = dataset.ADNI(PATHTODATASET, transform=transform)
testData = dataset.ADNI(PATHTODATASET, transform=transform, test=True)
trainLoader = DataLoader(trainData, batch_size=128, shuffle=True)
testLoader = DataLoader(testData, batch_size=128, shuffle=False)


