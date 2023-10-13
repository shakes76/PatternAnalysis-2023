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


model_save_path = '/content/drive/MyDrive/siamese_network.pth'

for epoch in range(Config.train_number_epochs):
    for i, data in enumerate(train_dataloader, 0):
        img0, img1, label = data
        img0, img1, label = img0, img1, label
        optimizer.zero_grad()
        output1, output2 = net(img0, img1)
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch, loss_contrastive.item()))
            iteration_number += 10
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

# Save the trained model to a .pth file
torch.save(net.state_dict(), model_save_path)

# Visualize loss history if needed
plt.plot(iteration_counter, loss_history)
plt.xlabel('No of Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()