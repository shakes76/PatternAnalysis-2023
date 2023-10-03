import os
from torchvision.io import read_image
import torch

#turns the ADNI raw data into a pytorch dataset to be used with a dataloader

class ADNI(torch.utils.data.Dataset):
    #1076 unique patients, 20 scans each, so take top 216 patients as the validation set, 108 healthy, 108 alzheimers
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
            self.validationData = []
            validationPatients = [""]*216
            for i, imageset in enumerate([AD, NC]):
                count = 0
                for image in imageset:
                    patientNumber = image.split("_")[0]
                    if patientNumber not in validationPatients and count != 108:
                        validationPatients[count * (i+1)] = patientNumber
                        count += 1

                    if patientNumber in validationPatients:
                        self.validationData.append(image)
                        self.data.pop(self.data.index(image))

                        if imageset == AD:
                            self.lastAD -= 1

            if self.isVal:
                self.lastAD = 108*20 - 1
            print(count)



                
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
    dataset = ADNI(path, transform=transform)

    allImages = torch.cat([dataset.__getitem__(i)[0].to(device) for i in range(len(dataset))])

    print(allImages)
    print(allImages.size())

    return allImages.mean(dtype=torch.float32), allImages.float().std()
#cropped outputs for a 0-255 format image (tensor(41.0344, device='cuda:0'), tensor(64.2557, device='cuda:0'))


