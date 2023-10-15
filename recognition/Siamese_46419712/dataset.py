import torchvision
import torch
import torchvision.transforms as transforms
import random
import numpy as np

TRAIN_PATH = "/home/groups/comp3710/ADNI/AD_NC/train"
# TRAIN_PATH = "./AD_NC/train"

TEST_PATH = "/home/groups/comp3710/ADNI/AD_NC/train"
# TEST_PATH = "./AD_NC/train"

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, trainset):
        # follow this source: https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
        self.trainset = trainset

    def __getitem__(self, index):
        img0, label0 = self.trainset[index]

        check_same_class = random.randint(0,1) 
        
        while True:
            img1, label1 = random.choice(self.trainset)

            if not torch.equal(img0, img1):
                if check_same_class and label0 == label1:
                    break
                elif not check_same_class and label0 != label1:
                    break
        
        return img0, img1, torch.from_numpy(np.array([int(label0 != label1)], dtype=np.float32))
    
    def __len__(self):
        return len(self.trainset)

class LoadData():
    def __init__(self, train=True, siamese=True):
        torch.manual_seed(40)
        random.seed(40)
        self.train = train
        self.siamese = siamese

        self.image_size = 105
        self.batch_size = 128
        self.num_worker = 0

        self.train_ratio = 0.8
        self.val_ratio = 1 - self.train_ratio

    def split_dataset(self, dataset, seed=True):
        if seed:
            generator = torch.Generator().manual_seed(40)
        else:
            generator = torch.Generator()

        # follow the principle -> 80% train 20% val
        train_set, val_set = torch.utils.data.random_split(dataset, [self.train_ratio, self.val_ratio], generator=generator)

        return train_set, val_set

    def load_data(self):
        if self.train:
            path = TRAIN_PATH
            transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])
        else:
            path = TEST_PATH
            transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor()
        ])

        image = torchvision.datasets.ImageFolder(root=path, transform=transform)
        train_image, val_image = self.split_dataset(image)

        if self.train:
            if self.siamese:
                trainset = PairedDataset(train_image)
                train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)

                valset = PairedDataset(val_image)
                val_loader = torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
                return train_loader, val_loader
            else:
                train_loader = torch.utils.data.DataLoader(train_image, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
                val_loader = torch.utils.data.DataLoader(val_image, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
                return train_loader, val_loader
        else:
            test_loader = torch.utils.data.DataLoader(image, batch_size=self.batch_size, shuffle=True, num_workers=self.num_worker)
            return test_loader

if __name__ == '__main__':
    # test dataset
    print("Start test load data")
    load_train_siamese, load_val_siamese = LoadData(train=True, siamese=True).load_data()
    load_train_classifier, load_val_classifier = LoadData(train=True, siamese=False).load_data()
    load_test = LoadData(train=False).load_data()
    print("Finish test load data")
