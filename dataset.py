# dataset.py
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import cv2

class SegmentationDataset(Dataset):
    def __init__(self, imageDir, maskDir, transforms=None, cache=False):
        # store the image and mask filepaths, and augmentation transforms
        self.cache = cache
        self.imagePaths = sorted(glob.glob(imageDir + "/*"))
        self.maskPaths = sorted(glob.glob(maskDir + "/*"))
        self.transforms = transforms
        if self.cache:
            self.cache_storage = [None] * self.__len__()

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # ... [Rest of the implementation]
        pass

    def get_dataloaders(batch_size=32):
        p = [transforms.Compose([transforms.ToTensor(), transforms.Resize((572,572))]),
            transforms.Compose([transforms.ToTensor(), transforms.Resize((388,388))])]

        train_dataset = SegmentationDataset("/content/drive/MyDrive/isic/isic-512/resized_train",
                                            "/content/drive/MyDrive/isic/isic-512/resized_train_gt",
                                            transforms=p, cache=True)

        test_dataset = SegmentationDataset("/content/drive/MyDrive/isic/isic-512/resized_test",
                                          "/content/drive/MyDrive/isic/isic-512/resized_test_gt",
                                          transforms=p, cache=True)

        valid_dataset = SegmentationDataset("/content/drive/MyDrive/isic/isic-512/resized_valid",
                                            "/content/drive/MyDrive/isic/isic-512/resized_valid_gt",
                                            transforms=p, cache=True)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        return train_dataloader, test_dataloader, valid_dataloader
