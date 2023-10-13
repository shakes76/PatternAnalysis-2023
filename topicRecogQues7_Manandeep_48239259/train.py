import torch
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt

class Config():
    training_dir = "/content/drive/MyDrive/training"
    testing_dir = "/content/drive/MyDrive/test"
    train_batch_size = 64
    train_number_epochs = 10
    folder_dataset = dset.ImageFolder(root=Config.training_dir)
    siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ])
                                       ,should_invert=False)
    train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=Config.train_batch_size)
    net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
counter = []
loss_history = []
iteration_number= 0