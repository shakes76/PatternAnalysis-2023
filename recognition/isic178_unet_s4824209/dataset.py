from torch.utils.data import Dataset
import os
from PIL import Image
from sklearn.model_selection import train_test_split


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

    img_path = sorted(os.listdir(img_root))
    gt_path = sorted(os.listdir(gt_root))

    images_train = []
    gt_train = []
    images_test = []
    gt_test = []

    #create a list of indices in the data
    indices = list(range(len(img_path)))

    #Split indices into random sets of 80/20
    train_indices, test_indices = train_test_split(indices, train_size=0.8, test_size=0.2)

    if mode == 'Train':
        for idx in train_indices:
            #Removes the txt files in the dataset
            if img_path[idx] == 'ATTRIBUTION.txt' or img_path[idx] == 'LICENSE.txt':
                continue
            image = os.path.join(img_root, img_path[idx])
            gt = os.path.join(gt_root, gt_path[idx])

            images_train.append(image)
            gt_train.append(gt)

        for idx in test_indices:
            if img_path[idx] == 'ATTRIBUTION.txt' or img_path[idx] == 'LICENSE.txt':
                continue
            image = os.path.join(img_root, img_path[idx])
            gt = os.path.join(gt_root, gt_path[idx])

            images_test.append(image)
            gt_test.append(gt)

        return images_train, gt_train, images_test, gt_test
    
    if mode == 'Test':

        for idx in range(len(img_path)):
            
            #removes txt files from dataset
            if img_path[idx] == 'ATTRIBUTION.txt' or img_path[idx] == 'LICENSE.txt':
                continue
            
            image = os.path.join(img_root, img_path[idx])
            gt = os.path.join(gt_root, gt_path[idx])

            images_test.append(image)
            gt_test.append(gt)

        return images_test, gt_test
