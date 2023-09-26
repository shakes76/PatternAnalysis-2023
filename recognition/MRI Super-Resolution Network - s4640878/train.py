import time
import torch, torchvision

from dataset import Dataset, machine
from modules import Model


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    """ training params """
    lr = 1e-3
    epochs = 100 if machine == "rangpur" else 3

    """ load training dataset """
    train_loader = Dataset(train=True).loader()

    """ model """
    model = Model().to(device)
    print(f"params: {sum([p.nelement() for p in model.parameters()])}", flush=True)

    """ loss function """
    loss_function = torch.nn.MSELoss()

    """ optimizer """
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4,
    )

    """ variable learning rate scheduller """
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs,
    )

    """ train the model """
    model.train()
    start = time.time()
    print("training...", flush=True)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            downsampled = torchvision.transforms.Resize(60, antialias=True)(images).to(device)

            """ forward pass """
            outputs = model(downsampled)
            loss = loss_function(outputs, images)

            """ backwards pass """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ print loss results """
            if (i + 1) % 300 == 0 or ((i + 1) % 10 == 0 and machine == "local"):
                print (
                    f"epoch: {epoch + 1}/{epochs}, step: {i + 1}/{len(train_loader)}, loss: {round(loss.item(), 5)}",
                    flush=True,
                )

            """ step the lr scheduler """
            lr_scheduler.step()

        """ save the model """
        if (epoch + 1) % 10 == 0 or (machine == "local" and epoch + 1 == epochs):
            with open(
                file="models/sr_model_%03d.pt" % (epoch + 1), mode="wb") as f:
                torch.save(obj=model.state_dict(), f=f)

    print(f"training time: {round((time.time() - start) / 60)} mins", flush=True)


if __name__ == "__main__":
    main()
