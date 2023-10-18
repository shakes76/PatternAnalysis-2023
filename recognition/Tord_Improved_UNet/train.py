import torch
from dataset import DiceLoss, load
from modules import UNet3D
from torch.utils.data import DataLoader

#training, validating, testing and saving the model

dataset = load()
dataLoader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = UNet3D(3,16).to(device)

criterion = DiceLoss()
lr_init = 5e-4
weight_decay = 1e-5
optimizer = torch.optim.Adam(net.parameters(), lr=lr_init, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_init * (0.985 ** epoch))


#Training the Network
for epoch in range(100):  

    running_loss = 0.0
    for i, data in enumerate(dataLoader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #loss = dice_coefficient(outputs, labels)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0
    scheduler.step()
    print('Finished Training epoch ', epoch + 1)
    torch.save(net.state_dict(), '~/recognition/Tord_Improved_UNet/model.pth')