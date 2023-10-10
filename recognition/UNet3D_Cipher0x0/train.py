from dataset import *
from modules import *

import sys

path = '*/file.txt'
sys.stdout = open(path, 'w')

epoch = 200
unet = UNet(1,6).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(unet.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=0.1, steps_per_epoch=len(dataloader) // 1 + 1, epochs=epoch)

unet.train()
loss_info = []
for epoch in range(epoch):
    for i, (inputs, labels) in enumerate(dataloader):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = unet(inputs.float().cuda())
        loss = criterion(outputs, labels.cuda())
        loss.backward()
        optimizer.step()

    print('[Epoch:'+ str(epoch) +'] [Loss:' + str(loss) + "]")
    sys.stdout.flush()

torch.save(unet, "*/unet3d.pth")