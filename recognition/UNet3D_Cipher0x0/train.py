from dataset import *
from modules import *


def main():
    # change path to current directory
    os.chdir(os.path.dirname(__file__))

    # check whether gpu is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device, flush=True)
    torch.cuda.empty_cache()

    # build model and optimizer
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())

    epoch = 30

    # train loop
    for i in range(epoch):
        unet.train()
        for index, data in enumerate(trainloader, 0):
            image, mask = data
            image, mask = ag.crop_and_augment(image, mask)
            image = image.unsqueeze(0)
            image = image.float().to(device)
            mask = mask.long().to(device)
            optimizer.zero_grad()
            pred = unet(image)
            loss = loss_fn(pred, mask)
            loss.backward()
            optimizer.step()

        print('One Epoch Finished', flush=True)
        torch.save(unet.state_dict(), 'net_paras.pth')


if __name__ == "__main__":
    main()
