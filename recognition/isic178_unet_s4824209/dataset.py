from torch.utils.data import Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import Train_Transform, Test_Transform



class customDataset(Dataset):
    '''
    Class that creates a custom dataset

    Args:
        images (List): list of paths to images 
        GT  (List): list of paths to ground truth 
        
        **These lists should should be same lenght

        ***The indexes of these two lists must match. Example: the ground truth of image at images[4]
            must be found at GT[4]

        transform (Transform objects): Transformations to perform on images and ground_truth.
            **The input of the transform must be a tuple
            
            ***The transformation will be applied to both image and ground_truth            
    
    '''

    def __init__(self, images, GT,transform):
        super(customDataset, self).__init__()
        self.images = images
        self.GT = GT
        self.transform = transform

    def __len__(self):
        '''
            returns lenght of the set
        '''
        if len(self.images) == len(self.GT):
            return(len(self.images))

    def __getitem__(self, idx):    
        img = Image.open(self.images[idx])
        GT = Image.open(self.GT[idx])

        #Applies the same transform to both image and corresponing ground truth
        if self.transform:
            img, GT = self.transform((img, GT))
                 
        return img, GT
    

def data_sorter(img_root, gt_root, mode):
    '''
    Function to sort the data into training and test sets

    args:
        img_root (String): path to the root folder of images
        gt_root (String): path to root folder of corresponding ground truth images

        mode: (Sting): 'Train'/'Tese', determines if the set is split into train/validate (80/20), or kept as a single test set
        

    returns:
    mode=train
        tuple
            images_train(List): List of paths to train images (String)
            gt_train(List): List of paths to train ground truth images (String)
            images_validation(List): List of paths to test images (String)
            gt_validation(List): List of paths to test gt images (String)
    
    mode=Test
        tuple
            images_test: list of path to images
            gt_test:  list of path to ground truth

    '''
    #Create identically sorted lists of filenames of img and gt
    img_path = sorted(os.listdir(img_root))
    gt_path = sorted(os.listdir(gt_root))

    #Lists used for relative paths to images and gt
    images_train = []
    gt_train = []
    images_test = []
    gt_test = []

    #create a list of one indice for each element in image/gt folder
    #if the list has 50 elements, the indices = [0,1,2.........49]
    indices = list(range(len(img_path)))

    #Create two lists with 80%/20% of the indices 
    train_indices, test_indices = train_test_split(indices, train_size=0.8, test_size=0.2)
    
    if mode == 'Train':
        #add train indices to train list
        for idx in train_indices:
            #Removes the txt files in the dataset
            if img_path[idx] == 'ATTRIBUTION.txt' or img_path[idx] == 'LICENSE.txt':
                continue
            #combine the root path, and the image/gt filename
            image = os.path.join(img_root, img_path[idx])
            gt = os.path.join(gt_root, gt_path[idx])

            images_train.append(image)
            gt_train.append(gt)
        
        #add test indices to test list
        for idx in test_indices:
            if img_path[idx] == 'ATTRIBUTION.txt' or img_path[idx] == 'LICENSE.txt':
                continue
            image = os.path.join(img_root, img_path[idx])
            gt = os.path.join(gt_root, gt_path[idx])

            images_test.append(image)
            gt_test.append(gt)

        return images_train, gt_train, images_test, gt_test
    
    if mode == 'Test':
        #add every element to test list (the data is not split)
        for idx in range(len(img_path)):
            
            #removes txt files from dataset
            if img_path[idx] == 'ATTRIBUTION.txt' or img_path[idx] == 'LICENSE.txt':
                continue
            
            image = os.path.join(img_root, img_path[idx])
            gt = os.path.join(gt_root, gt_path[idx])

            images_test.append(image)
            gt_test.append(gt)

        return images_test, gt_test

#Set batch size for trainloader
batch_size = 2

#LOAD DATA
#root path of test set (images and ground truth)
train_img_root = 'data'
train_gt_root = 'GT_data'

#root path of validation set
test_img_root ='validation_data/ISIC2018_Task1-2_Validation_Input'
test_gt_root = 'validation_data/ISIC2018_Task1_Validation_GroundTruth'

#Creating sorted lists of image and ground truth path for train and validation (80%/20% split)
img_train_path,gt_train_path, img_val_path, gt_val_path = data_sorter(img_root=train_img_root, gt_root=train_gt_root, mode='Train')

#Defining transforms for trainset and testset
train_transform = transforms.Compose([Train_Transform()])
test_transform = transforms.Compose([Test_Transform()])

#Create the trainset and loading into dataloader with defined transforms
train_set = customDataset(images=img_train_path, GT=gt_train_path, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

#Create the validation-set and loading into dataloader
#Validation loader has batch_size=1 to be able to check dice score of each separate image 
validation_set = customDataset(images=img_val_path, GT=gt_val_path, transform=test_transform)
validation_loader = DataLoader(validation_set, batch_size=1)


#Create the test-set dataloader
#Test loader has batch_size=1 to be able to check dice score of each separate image
img_test_path,gt_test_path = data_sorter(img_root=test_img_root, gt_root=test_gt_root, mode='Test')
test_set = customDataset(images=img_test_path, GT=gt_test_path, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=1)