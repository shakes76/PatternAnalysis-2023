import torch
from torch.utils.data import DataLoader
from torch import optim
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch
from modules import CustomSiameseNetwork, ContrastiveLoss
from dataset import CustomDataset


class Config():
    training_dir = "/content/AD_NC/train"
    testing_dir = "/content/AD_NC/test"
    batch_size = 32
    num_epochs = 6

# Initialize the dataset and data loader
folder_dataset = datasets.ImageFolder(root=Config.training_dir)
custom_dataset = CustomDataset(image_dataset=folder_dataset,
                               transform=transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()]),
                               should_invert=False)

train_data_loader = DataLoader(custom_dataset,
                               shuffle=True,
                               num_workers=8,
                               batch_size=Config.batch_size)

# Initialize the Siamese network, loss criterion, and optimizer
siamese_net = CustomSiameseNetwork()
loss_criterion = ContrastiveLoss()
optimizer = optim.Adam(siamese_net.parameters(), lr=0.0005)

# Lists to store loss and iteration data for visualization
iteration_counter = []
loss_history = []
iteration_number = 0

model_save_path = '/content/drive/MyDrive/dataset/model.pth'
count = 1

# Training loop
for epoch in range(Config.num_epochs):
    for i, data in enumerate(train_data_loader, 0):
        img1, img2, label = data
        optimizer.zero_grad()
        output1, output2 = siamese_net(img1, img2)
        loss_contrastive = loss_criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch {} - Iteration {} - Loss {}".format(epoch + 1, count, loss_contrastive.item()))
            count += 1
            iteration_number += 10
            iteration_counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

    
    if epoch + 1 == Config.num_epochs:
        break 


torch.save(siamese_net.state_dict(), model_save_path)


plt.plot(iteration_counter, loss_history)
plt.xlabel('No of Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
