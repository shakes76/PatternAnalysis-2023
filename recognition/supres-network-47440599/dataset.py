import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    dataroot = "./data/AD_NC/train"
    downscaling = 4

    batch_size = 128

    workers = 2

    dataset = dset.ImageFolder(root=dataroot,
                            transform=transforms.Compose([
                                transforms.Grayscale(1),
                                transforms.Resize((64,60)),
                                transforms.ToTensor(),
                                transforms.RandomHorizontalFlip()
                            ]))

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    print("data loaded")
    return dataloader


    

if __name__ == "__main__":
    dataloader = load_data()
    device = torch.device("cpu")
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)),cmap = "gray")
    plt.show()