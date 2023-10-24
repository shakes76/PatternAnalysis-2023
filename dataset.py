import numpy as np
import torch
from PIL import Image
import os
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split

transform = transforms.Compose([
    transforms.PILToTensor()
])

xtrain = []
xtest = []
ytrain = []
ytest = []
slicemax = 20 #20 images per patient

ntrainimgs_AD = 0
patient = []
slice = 0
for filename in sorted(os.listdir('../ADNI_AD_NC_2D/AD_NC/train/AD/')):
    f = os.path.join('../ADNI_AD_NC_2D/AD_NC/train/AD/', filename)
    img = Image.open(f)
    imgtorch = transform(img).float()
    imgtorch.require_grad = True
    patient.append(imgtorch/255) #go from 0,255 to 0,1
    slice  = (slice+1) % slicemax
    if slice == 0:
        xtrain.append(torch.stack(patient))
        patient = []
        ntrainimgs_AD += 1
    pass  
ntrainimgs_NC = 0
patient = []
slice = 0
for filename in sorted(os.listdir('../ADNI_AD_NC_2D/AD_NC/train/NC')):
    f = os.path.join('../ADNI_AD_NC_2D/AD_NC/train/NC', filename)
    img = Image.open(f)
    imgtorch = transform(img).float()
    imgtorch.require_grad = True
    patient.append(imgtorch/255) #go from 0,255 to 0,1
    slice  = (slice+1) % slicemax
    if slice == 0:
        xtrain.append(torch.stack(patient))
        patient = []
        ntrainimgs_NC += 1
    pass   
ntestimgs_AD = 0
patient = []
slice = 0
for filename in sorted(os.listdir('../ADNI_AD_NC_2D/AD_NC/test/AD')):
    f = os.path.join('../ADNI_AD_NC_2D/AD_NC/test/AD', filename)
    img = Image.open(f)
    imgtorch = transform(img).float()
    imgtorch.require_grad = True
    patient.append(imgtorch/255) #go from 0,255 to 0,1
    slice  = (slice+1) % slicemax
    if slice == 0:
        xtest.append(torch.stack(patient))
        patient = []
        ntestimgs_AD += 1
    pass   
ntestimgs_NC = 0
patient = []
slice = 0
for filename in sorted(os.listdir('../ADNI_AD_NC_2D/AD_NC/test/NC')):
    f = os.path.join('../ADNI_AD_NC_2D/AD_NC/test/NC', filename)
    img = Image.open(f)
    imgtorch = transform(img).float()
    imgtorch.require_grad = True
    patient.append(imgtorch/255) #go from 0,255 to 0,1
    slice  = (slice+1) % slicemax
    if slice == 0:
        xtest.append(torch.stack(patient))
        patient = []
        ntestimgs_NC += 1
    pass    
xtrain = torch.stack(xtrain)
xtest = torch.stack(xtest)
ytrain = torch.from_numpy(np.concatenate((np.ones(ntrainimgs_AD), np.zeros(ntrainimgs_NC)), axis=0))
ytest = torch.from_numpy(np.concatenate((np.ones(ntestimgs_NC), np.zeros(ntestimgs_NC)), axis=0))

data_val, data_test = random_split(TensorDataset(xtest, ytest), [0.7,0.3])
dataloader_train = DataLoader(TensorDataset(xtrain, ytrain), batch_size=32, shuffle=True)
dataloader_test = DataLoader(data_test, batch_size=32, shuffle=True)
dataloader_val = DataLaoder(data_val, batch_size=32, shuffle=True)

def returnDataLoaders():
    return dataloader_train, dataloader_test, dataloader_val