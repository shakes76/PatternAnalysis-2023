import torchvision
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import matplotlib.pyplot as plt


class PairedDataset(torch.utils.data.Dataset):
    
    def __init__(self, trainset):
        # follow this source: https://datahacker.rs/019-siamese-network-in-pytorch-with-application-to-face-similarity/
        self.trainset = trainset

    def __getitem__(self, index):
        img0, label0 = self.trainset[index]

        check_same_class = random.randint(0,1) 
        if check_same_class:
            while True:
                #Look untill the same class image is found
                img1, label1 = random.choice(self.trainset)
                if torch.equal(img0, img1):
                    continue

                if label1 == label0:
                    break
        else:
            while True:
                #Look untill a different class image is found
                img1, label1 = random.choice(self.trainset)
                
                if torch.equal(img0, img1):
                    continue
                
                if label1 != label0:
                    break
        
        return img0, img1, torch.from_numpy(np.array([int(label0 != label1)], dtype=np.float32))
    
    def __len__(self):
        return len(self.trainset)

def load_train_data():
    path="/home/groups/comp3710/ADNI/AD_NC/train"
    # this for testing
    path="./AD_NC/train"

    # Data transformation
    transform_train = transforms.Compose([
        transforms.Resize(105),
        transforms.CenterCrop(105),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # transform_train = transforms.Compose([
    #     transforms.Resize(105),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.CenterCrop(105),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # ])
    train_image = torchvision.datasets.ImageFolder(root=path, transform=transform_train)
    trainset = PairedDataset(train_image)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

    # data_iter = iter(train_loader)
    # images, _ = next(data_iter)

    # # Check the number of channels in the first image
    # num_channels = images.size(1)
    # print(f"Number of channels in the first image: {num_channels}")

    return train_loader

def load_test_data():
    path="/home/groups/comp3710/ADNI/AD_NC/test"
    # this for testing
    path="./AD_NC/test"
    # dataset = torchvision.datasets.ImageFolder(root=path, transform=transforms.ToTensor())

    # # Calculate the mean and standard deviation
    # mean = torch.mean(dataset.data.float() / 255.0)
    # std = torch.std(dataset.data.float() / 255.0)

    # Data transformation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # standard scaling for normalize, doesn't know much on the status of the entire dataset
    ])

    test_image = torchvision.datasets.ImageFolder(root=path, transform=transform_test)
    testset = PairedDataset(test_image)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=True, num_workers=0)
    return test_loader

if __name__ == '__main__':
    # test dataset

    testloader = load_train_data()
        
    # # Extract one batch
    # example_batch = next(iter(testloader))

    # # Example batch is a list containing 2x8 images, indexes 0 and 1, an also the label
    # # If the label is 1, it means that it is not the same person, label is 0, same person in both images
    # concatenated = torch.cat((example_batch[0], example_batch[1]),0)

    # npimg = torchvision.utils.make_grid(concatenated).numpy()
    # plt.axis("off")

    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.show() 
    # print(example_batch[2].numpy().reshape(-1))
