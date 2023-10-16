import os
from PIL import Image
import torchvision.transforms as transforms
from modules import CustomSiameseNetwork
from dataset import CustomDataset
import torchvision.datasets as dset
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import PIL.ImageOps
import torch.nn.functional as F
import torch.nn.functional as TorchFun
import torchvision
import matplotlib.pyplot as plt
import numpy np
from PIL import Image
from torch.autograd import Variable

def imshow(img, text=None, should_save=False):
    npimg = np.array(Image.open(img))
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
            bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})
    plt.imshow(npimg)
    plt.show()

def imshow_grid(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0))
    plt.show()

trained_siamese_net = CustomSiameseNetwork()
trained_siamese_net.load_state_dict(torch.load('/content/drive/MyDrive/dataset/model.pth'))
trained_siamese_net.eval()

# Define the test dataset
testing_dir = '/content/AD_NC/test'
folder_dataset_test = dset.ImageFolder(root=testing_dir)
siamese_dataset = CustomDataset(folder_dataset_test,
                                transform=transforms.Compose([transforms.Resize((100, 100)),
                                                              transforms.ToTensor()
                                                              ]),
                                should_invert=False)

# Define the data loader for testing
test_dataloader = DataLoader(siamese_dataset, num_workers=6, batch_size=1, shuffle=True)
dataiter = iter(test_dataloader)
x0, _, _ = next(dataiter)
for i in range(10):
    _, x1, label2 = next(dataiter)
    print(label2)
    concatenated = torch.cat((x0, x1), 0)

    output1, output2 = trained_siamese_net(Variable(x0), Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    imshow_grid(torchvision.utils.make_grid(concatenated),'Dissimilarity: {:.2f}'.format(euclidean_distance.item()))
