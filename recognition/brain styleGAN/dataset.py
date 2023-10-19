import torch
import torchvision
import matplotlib.pyplot as plt

"""
Contains the data loader for loading and preprocessing your data
"""

#root directory for dataset
dataroot = "C:/Users/aleki/Desktop/Lab report/ADNI_AD_NC_2D/AD_NC/train"
labels_map = {
    0: "AD",
    1: "NC"
}
image_size = 128
batch_size = 16

def load_dataset():
    #set transform
    transform=torchvision.transforms.Compose([
                                            torchvision.transforms.Resize(image_size),
                                            torchvision.transforms.CenterCrop(image_size),
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize((0.5), (0.5)),
                                            ])
    #load dataset
    training_data = torchvision.datasets.ImageFolder(root=dataroot, transform=transform)

    #create the dataloader
    dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

    return dataloader, training_data

def check_loader():
    loader, training_data = load_dataset()
    plt.suptitle("samples")
    for i in range(9):
        rand_index = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[rand_index]
        plt.subplot(3, 3, i+1)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow((img.permute(1,2,0)+1)/2) #rearange dimensions to match rgb image format

check_loader()
plt.show()