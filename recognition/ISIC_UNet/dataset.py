import os
from PIL import Image
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, root_dir, transform): 
        super(CustomDataset,self).__init__()
        self.root_dir = root_dir                            #Path for image folder
        self.transform = transform                          #Transform for resize and converting to tensor
        self.images = sorted(os.listdir(root_dir))          #Extract images in order
        self.images.remove('ATTRIBUTION.txt','LICENSE:txt') #Removes non image files
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.images[idx]) 
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        return image
