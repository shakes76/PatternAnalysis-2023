import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.datasets as datasets
import torch.utils.data as util_data
from torch.utils.data import DataLoader, Dataset
import torch
import random
from PIL import Image
import dataset as ds
import module as md


if __name__ == '__main__':

    def imshow(img, text=None):
        npimg = img.numpy()
        plt.axis("off")
        if text:
            plt.text(75, 8, text, style='italic',fontweight='bold',
                bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
            
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    train_path = "C:\\Users\\Asus\\Desktop\\AD_NC\\train"
    test_path = "C:\\Users\\Asus\\Desktop\\AD_NC\\test"

    training_dataset = datasets.ImageFolder(root=train_path)
    testing_dataset = datasets.ImageFolder(root=test_path)
    transform = transforms.Compose([transforms.ToTensor()])

    siamese_train = ds.Siamese_dataset(imageFolder=training_dataset, transform=transform)
    siamese_test = ds.Siamese_dataset(imageFolder=testing_dataset, transform=transform)

    vis_dataloader = DataLoader(siamese_train,
                            shuffle=True,
                            num_workers=2,
                            batch_size=8)


    # Run this to test your data loader
    image1, image2, label = next(iter(vis_dataloader))

    concatenated = torch.cat((image1, image2),0)
    imshow(torchvision.utils.make_grid(concatenated))
    print(label.numpy().reshape(-1))

