import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

'''
This file reads a downloaded image dataset from the OASIS dataset and returns a data loader of it 
along with a transformation of the original dataset;
Additionally, this file gives some sample images from the loader to show what the images I'm working
with look like.
'''

DATASET1 = "/home/groups/comp3710/OASIS/keras_png_slices_train" # training data
DATASET2 = "/home/groups/comp3710/OASIS/keras_png_slices_test" # test data
DATASET3 = "/home/groups/comp3710/OASIS/keras_png_slices_validate" # validation data
BATCH_SIZES = [256, 128, 64, 32, 16, 8]
CHANNELS_IMG = 3

# Costomized ImageFolder to read image data
# Reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
class CustomImageDataset(Dataset):
    def __init__(self, img_dirs, transform=None):
        #self.img_dir = img_dir
        #self.img_files = os.listdir(img_dir)
        self.transform = transform

        # Read all the three OASIS files from rangpur
        self.img_files = []
        for dir_path in img_dirs:
            self.img_files += [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        #img_name = os.path.join(self.img_dir, self.img_files[idx])
        img_name = self.img_files[idx]
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def get_loader(image_size):
    trainsform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), # not necessary
         transforms.ToTensor(),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)],
         )
        ]
    )
    #batch_size = BATCH_SIZES[int(log2(image_size/4))] # image size = 256 
    batch_size = BATCH_SIZES[4] # img size = 256, batch size = 16
     # Load all the training, test, and validation data together to train the styleGAN model
    dataset = CustomImageDataset(img_dirs = [DATASET1, DATASET2, DATASET3], transform=trainsform)
    loader = DataLoader(dataset,batch_size=batch_size,shuffle=True)
    return loader, dataset

# shape (256, 256, 3)

def check_loader():
    loader,_ = get_loader(256)
    img = next(iter(loader))
    _,ax = plt.subplots(3,3,figsize=(8,8))
    plt.suptitle('Real sample images')
    ind = 0
    for k in range(3):
        for kk in range(3):
            ax[k][kk].imshow((img[ind].permute(1,2,0)+1)/2)
            ind +=1

    if not os.path.exists("output_images"):
        os.makedirs("output_images")

    # Save the figure to the specified path
    save_path = os.path.join("output_images", "real_grid.png")
    plt.savefig(save_path)

    plt.close()
            
check_loader() 
# Sample gird shown in the "output_images" folder in the same directory


