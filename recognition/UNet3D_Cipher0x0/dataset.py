import glob
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class NiiImageLoader(DataLoader):
    def __init__(self, image_path, mask_path):
        self.inputs = []
        self.masks = []
        # retrieve path from dataset
        for f in sorted(glob.iglob(image_path)):
            self.inputs.append(f)
        for f in sorted(glob.iglob(mask_path)):
            self.masks.append(f)
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.inputs)

    # open files
    def __getitem__(self, idx):
        image_p = self.inputs[idx]
        mask_p = self.masks[idx]

        image = nibabel.load(image_p)
        image = np.asarray(image.dataobj)

        mask = nibabel.load(mask_p)
        mask = np.asarray(mask.dataobj)

        image = self.to_tensor(image)
        image = image.unsqueeze(0)
        image = image.data

        mask = self.to_tensor(mask)
        mask = mask.unsqueeze(0)
        mask = mask.data

        return image, mask


# # load the dataset
dataset = NiiImageLoader("/home/groups/comp3710/HipMRI_Study_open/semantic_MRs/*",
                         "/home/groups/comp3710/HipMRI_Study_open/semantic_labels_only/*")

# split the dataset
trainloader, valloader, testloader = torch.utils.data.random_split(dataset, [179, 16, 16])
