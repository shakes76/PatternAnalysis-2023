import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from math import log2
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

DATASET = "/home/groups/comp3710/OASIS/keras_png_slices_train"
BATCH_SIZES = [256, 128, 64, 32, 16, 8]
CHANNELS_IMG = 3

# Costomized ImageFolder to read image data
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.img_files = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_files[idx])
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
    batch_size = BATCH_SIZES[0] # batch size = 256
    dataset = CustomImageDataset(img_dir = DATASET, transform=trainsform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return loader, dataset

# shape (256, 256, 3)

def check_loader():
    loader,_ = get_loader(256)
    img  = next(iter(loader))
    _,ax     = plt.subplots(3,3,figsize=(8,8))
    plt.suptitle('Some real samples')
    ind = 0
    for k in range(3):
        for kk in range(3):
            ax[k][kk].imshow((img[ind].permute(1,2,0)+1)/2)
            ind +=1

    if not os.path.exists("output_images"):
        os.makedirs("output_images")

    # Save the figure to the specified path
    save_path = os.path.join("output_images", "sample_grid.png")
    plt.savefig(save_path)

    plt.close()
            
check_loader() 
# Sample gird shown in the "output_images" folder in the same directory


