from dataset import *
from modules import *

import os


# Calculator for Dice similarity coefficient
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def single_loss(self, inputs, targets, smooth=0.1):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return dice

    def forward(self, inputs, targets):
        channel = 6
        # separate the prediction
        for i in range(channel):
            vars()["inputs" + str(i)] = (inputs.argmax(1) == i)

        # the target for each channel
        for i in range(channel):
            vars()["targets" + str(i)] = (targets == i)

        # calculate dsc for each channel
        for i in range(channel):
            vars()["dice" + str(i)] = 1 - self.single_loss(
                vars()["inputs" + str(i)], vars()["targets" + str(i)])

        # calculate average dsc
        dice_avg = (dice0 + dice1 + dice2 + dice3 + dice4 + dice5) / 6.0
        dice_list = [dice0, dice1, dice2, dice3, dice4, dice5, dice_avg]

        return dice_avg, dice_list


# main loop
def main():
    # setting path to current directory
    os.chdir(os.path.dirname(__file__))

    epoch = 30

    # build model, loss function and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(unet.parameters())

    # train loop
    for i in range(epoch):
        unet.train()
        for index, data in enumerate(trainloader, 0):
            image, mask = data
            image = image.unsqueeze(0).float().to(device)
            mask = mask.long().to(device)
            optimizer.zero_grad()
            pred = unet(image)
            loss = loss_fn(pred, mask)
            loss.backward()
            optimizer.step()

        print('Epoch ' + str(i) + ' Finished', flush=True)
        # evaluate the model every epoch
        unet.eval()
        num_batches = len(valloader)
        val_loss = 0
        dice_all = 0
        dice_detail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        with torch.no_grad():
            for X, y in valloader:
                X = X.unsqueeze(0).float().to(device)
                y = y.long().to(device)
                pred = unet(X)
                val_loss += loss_fn(pred, y).item()
                dsc, dsc_list = dice_loss(pred, y)
                dice_all += (1 - dsc)
                for j in range(6):
                    dice_detail[j] += (1 - dsc_list[j])

        val_loss /= num_batches
        dice_all /= num_batches
        for k in range(6):
            dice_detail[k] /= num_batches
        print(f"Avg loss: {val_loss:>8f}", flush=True)
        print(f"AvgValid DSC: {dice_all:>8f}", flush=True)
        print("Valid DSC:" + str(dice_detail), flush=True)
        torch.save(unet.state_dict(), 'net_paras.pth')

        # test the model every epoch
        unet.eval()
        num_batches = len(testloader)
        dice_all = 0
        dice_detail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        with torch.no_grad():
            for X, y in testloader:
                X = X.unsqueeze(0).float().to(device)
                y = y.long().to(device)
                pred = unet(X)
                dsc, dsc_list = dice_loss(pred, y)
                dice_all += (1 - dsc)
                for j in range(6):
                    dice_detail[j] += (1 - dsc_list[j])

        dice_all = dice_all / num_batches
        for k in range(6):
            dice_detail[k] /= num_batches
        print(f"AvgTest DSC: {dice_all:>8f} \n", flush=True)
        print("Test DSC:" + str(dice_detail), flush=True)


if __name__ == "__main__":
    main()
