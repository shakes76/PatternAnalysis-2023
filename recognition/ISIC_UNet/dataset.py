import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

SPLIT=0.05

#Class for loading data set and converting form images to tensors
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform): 
        super(CustomDataset,self).__init__()
        self.root_dir = root_dir                                                #Path for image folder
        self.transform = transform                                              #Transform for resize and converting to tensor
        self.images = sorted(os.listdir(root_dir))                              #Extract images in 
        if 'ATTRIBUTION.txt' in self.images and  'LICENSE.txt' in self.images:  #Removes non image files
            self.images.remove('ATTRIBUTION.txt')                              
            self.images.remove('LICENSE.txt')   

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx])  #Combines the path and the image name
        image = Image.open(img_name)                              #Loads the image
        if self.transform:                                        #Transformst the image
            image = self.transform(image)

        return image
    


#Function for image augmentation transformations for the training set.   
def transformation_train(trainset,truth_trainset):
    trainset_trans=[]                                            #New list for transfomations
    to_pil = transforms.ToPILImage()                             #Function for converting tensor to PIL image
    random_transform= RandomTransform(transform=transform_train) #Innit randomtransform class

    #Runs through the split data set lists. Converst back to PIL img and preforms transform on the images.
    for pic,truth in zip(trainset,truth_trainset):
        pic=to_pil(pic)
        truth=to_pil(truth)
        transformed=random_transform(pic,truth)
        trainset_trans.append(transformed)
    return trainset_trans
    

#Takes two elements and applies the same random transforms to both of them.
class RandomTransform:
    def __init__(self, transform):
        self.transform = transform          #Transform for trainset and truth

    def __call__(self, element1, element2):
        seed=random.randint(0, 1000)        #Create a random seed

        random.seed(seed)                   #Set random seed
        torch.manual_seed(seed)             #Set random seed for torch
        output1=self.transform(element1)    #Transform for normal images

        random.seed(seed)                   #Resets seed so the random transforms are the same for truth and normal
        torch.manual_seed(seed)
        output2=self.transform(element2)

        return (output1,output2)



##Data and transforms
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((512,512),antialias=True)]) #Transforms for resizing and to tensor

transform_train = transforms.Compose([                                                              #Transforms for training data normal images
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),                                                              #Randomly flips image
    transforms.RandomRotation(90),                                                                  #Randomly rotates images 90 degrees
    transforms.RandomResizedCrop(512, scale=(0.9, 1.5), ratio=(0.75, 1.333), interpolation=2),      #Randomly resized and crops the image
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=(-0.4,0.4))           #Randomly changes brightness, contrast, saturation and hue of image
])



data = CustomDataset(root_dir='/home/snorrebs/Pattern_3710/ass3_test/ISIC_UNet/pictures', transform=transform)                  #Loads data, resizes and converts to tensor.
trainset,validationset=train_test_split(data,test_size=SPLIT,shuffle=False)                                                     #Splits normal dataset into training and validationset 

truthset = CustomDataset(root_dir='/home/snorrebs/Pattern_3710/ass3_test/ISIC_UNet/pictures_seg', transform=transform)          #Loads truth data for training 
truth_trainset,truth_validationset=train_test_split(truthset,test_size=SPLIT,shuffle=False)                                     #Splits truth set into training and validation  

testset=CustomDataset(root_dir='/home/snorrebs/Pattern_3710/ass3_test/ISIC_UNet/ISIC2018_Task1-2_Validation_Input', transform=transform)           #Loads the data set
testset_truth=CustomDataset(root_dir='/home/snorrebs/Pattern_3710/ass3_test/ISIC_UNet/ISIC2018_Task1_Validation_GroundTruth', transform=transform) #Loads the truth for the testset 


trainset_trans=transformation_train(trainset,truth_trainset)            #Preforms image augmentation transforms on training data. See utils.py
 

#Uses data loader to create datasets
trainloader = DataLoader(trainset_trans, batch_size=3, shuffle=True)                    #Training set - normal images
validationloader=DataLoader(validationset, batch_size=3, shuffle=False)                 #Validation set - normal images
truth_validationset_loader=DataLoader(truth_validationset, batch_size=3, shuffle=False) #Validation set - truth images
testloader=DataLoader(testset, batch_size=1, shuffle=False)                             #Test set - normal images
testset_truth_loader=DataLoader(testset_truth, batch_size=1, shuffle=False)             #Test set - truth images
