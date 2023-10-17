from dataset import *
from modules import *

import os


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    '''calculate dsc per label'''

    def single_loss(self, inputs, targets, smooth=0.1):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return dice

    '''calculate dsc for each channel, add them up and get the mean'''

    def forward(self, inputs, targets):
        input0 = (inputs.argmax(1) == 0)  # prediction of label 0
        input1 = (inputs.argmax(1) == 1)  # prediction of label 1
        input2 = (inputs.argmax(1) == 2)  # prediction of label 2
        input3 = (inputs.argmax(1) == 3)  # prediction of label 3
        input4 = (inputs.argmax(1) == 4)  # prediction of label 4
        input5 = (inputs.argmax(1) == 5)  # prediction of label 5

        target0 = (targets == 0)  # target of label 0
        target1 = (targets == 1)  # target of label 1
        target2 = (targets == 2)  # target of label 2
        target3 = (targets == 3)  # target of label 3
        target4 = (targets == 4)  # target of label 4
        target5 = (targets == 5)  # target of label 5

        dice0 = 1 - self.single_loss(input0, target0)
        dice1 = 1 - self.single_loss(input1, target1)
        dice2 = 1 - self.single_loss(input2, target2)
        dice3 = 1 - self.single_loss(input3, target3)
        dice4 = 1 - self.single_loss(input4, target4)
        dice5 = 1 - self.single_loss(input5, target5)

        dice_avg = (dice0 + dice1 + dice2 + dice3 + dice4 + dice5) / 6.0

        dice_list = [dice0, dice1, dice2, dice3, dice4, dice5, dice_avg]

        return dice_avg, dice_list


def main():
    # change path to current directory
    os.chdir(os.path.dirname(__file__))

    epoch = 30

    # build model and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    dice_loss = DiceLoss()
    optimizer = torch.optim.Adam(unet.parameters())

    # train loop
    print("Training Begin")
    for i in range(epoch):
        unet.train()
        for index, data in enumerate(trainloader, 0):
            image, mask = data
            image = image.unsqueeze(0)
            image = image.float().to(device)
            mask = mask.long().to(device)
            optimizer.zero_grad()
            pred = unet(image)
            loss = loss_fn(pred, mask)
            loss.backward()
            optimizer.step(),

        # run the model on the val set after each train loop
        unet.eval()
        num_batches = len(valloader)
        val_loss = 0
        dice_all = 0
        dice_detail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        with torch.no_grad():
            for X, y in valloader:
                X = X.unsqueeze(0)
                X = X.float().to(device)
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

        print('Epoch' + str(i) + 'Finished', flush=True)
        print(f"Avg loss: {val_loss:>8f}", flush=True)
        print(f"AvgValid DSC: {dice_all:>8f}", flush=True)
        print("Valid DSC:" + str(dice_detail), flush=True)
        torch.save(unet.state_dict(), 'net_paras.pth')

        # run on test set after the train is finished
        unet.eval()
        num_batches = len(testloader)
        dice_all = 0
        dice_detail = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        with torch.no_grad():
            for X, y in testloader:
                X = X.unsqueeze(0)
                X = X.float().to(device)
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
