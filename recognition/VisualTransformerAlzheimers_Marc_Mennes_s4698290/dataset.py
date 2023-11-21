"""
Implements a pytorch Dataset type for the ADNI image data

Note the initialiser expects the data folder structure to be of the same
format given by the COMP3710 course page download, that being:

v path to the dataset
  v AD_NC
    v test
      v AD
        image1.jpeg
        image2.jpeg
        ...
      v NC
        image1.jpeg
        image2.jpeg
        ...
    v train
      v AD
        image1.jpeg
        image2.jpeg
        ...
      v NC
        image1.jpeg
        image2.jpeg
        ...

Where each image name is of the format: [PATIENT ID]_[MRI IMAGE NUMBER].jpeg
"""
import os
from torchvision.io import read_image
import torchvision
import torch

#turns the ADNI raw data into a pytorch dataset to be used with a dataloader
class ADNI(torch.utils.data.Dataset):
    #1076 unique patients, 20 scans each, so take 216 patients as the validation set, 108 healthy, 108 alzheimers
    def __init__(self, path, transform=None, test=False, validation=False):
        self.path = path #path not ending with a /
        self.transform = transform
        self.isTest = test
        self.isVal = validation

        #get a list of image names
        if not test:
            AD = os.listdir(self.path + "/AD_NC/train/AD")
            NC = os.listdir(self.path + "/AD_NC/train/NC")
        else:
            AD = os.listdir(self.path + "/AD_NC/test/AD")
            NC = os.listdir(self.path + "/AD_NC/test/NC")
        
        #map the images to indices for __getitem__
        self.data = AD + NC
        self.lastAD = len(AD) - 1

        if not self.isTest:

            #construct the validation set
            self.validationData = []
            validationPatients = []
            for i, imageset in enumerate([AD, NC]):
                count = 0
                #take 108 unique patient numbers and all of their scans
                for image in imageset:
                    patientNumber = image.split("_")[0]

                    #keep collecting new patient IDs till we have 108 of them
                    if patientNumber not in validationPatients and count < 108:
                        validationPatients.append(patientNumber)
                        count += 1

                    #if the patient IDs is collected, put the image in the validation set
                    #then remove it from the train set
                    if patientNumber in validationPatients:
                        self.validationData.append(image)
                        self.data.pop(self.data.index(image))

                        #keep track of the last alzheimers positive case in the dataset
                        if imageset == AD:
                            self.lastAD -= 1

            if self.isVal:
                self.lastAD = 108*20 - 1


    def __len__(self):
        if self.isVal and not self.isTest:
            return len(self.validationData)
        elif not self.isVal and not self.isTest:
            return len(self.data)
        else:
            return len(self.data)
         
    def __getitem__(self, idx):
        
        imageName = self.data[idx]
        if not self.isTest and self.isVal:
            imageName = self.validationData[idx]

        ADLabel = 0

        #build the path to the image and find the label
        if idx <= self.lastAD:
            pathToImage = self.path + "/AD_NC" + ("/test" if self.isTest else "/train") + "/AD/"
            ADLabel = 1
        else:
            pathToImage = self.path + "/AD_NC" + ("/test" if self.isTest else "/train") + "/NC/"
        
        image = read_image(pathToImage + imageName)

        if self.transform is not None:
            image = self.transform(image)

        return image, ADLabel #ADLabel = 1 means alzheimer's, 0 means normal
    
#calculate norm and std of the dataset
def find_mean_std(path, transform = None):
    device = torch.device("cuda")
    train = ADNI(path, transform=transform)
    val = ADNI(path, transform=transform, validation=True)
    test = ADNI(path, transform=transform, test = True)

    #load all the images into gpu memory in one torch tensor
    allImages = torch.cat([train.__getitem__(i)[0].to(device) for i in range(len(train))] +[test.__getitem__(i)[0].to(device) for i in range(len(test))] +[val.__getitem__(i)[0].to(device) for i in range(len(val))]  )

    print(allImages.size())

    return allImages.mean(dtype=torch.float32), allImages.float().std()

#dataset mean: 31.5226 dataset std: 58.8811

