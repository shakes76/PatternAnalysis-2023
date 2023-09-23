import sys, os, time
import torch, torchvision
from traceback import format_exc

from dataset import Dataset
from modules import ResNet18, ResNet34


def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    """ training params """
    lr = 1e-3
    epochs = 1

    """ load datasets """
    train_loader = Dataset(train=True).loader()
    test_loader = Dataset(train=False).loader()

    """ model """
    model = ResNet18().to(device)
    print(f"params: {sum([p.nelement() for p in model.parameters()])}", flush=True)

    """ loss function """
    loss_function = torch.nn.CrossEntropyLoss()

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

            """ forward pass """
            outputs = model(images)
            if i == 0: print(f"{outputs = }\n{labels = }", flush=True)
            loss = loss_function(outputs, labels)

            """ backwards pass """
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            """ print loss results """
            if (i + 1) % 10 == 0:
                print (
                    f"epoch: {epoch + 1}/{epochs}, step: {i + 1}/{len(train_loader)}, loss: {round(loss.item(), 5)}",
                    flush=True,
                )

            """ step the lr scheduler """
            lr_scheduler.step()

    print(f"training time: {round((time.time() - start) / 60)} mins", flush=True)

    """ save the model """
    with open(file=f"models/model_test.pt", mode="wb") as f:
        torch.save(obj=model.state_dict(), f=f)


    """ test the model """
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predicted = torch.max(outputs.data, 1)[1]
            expected = torch.max(labels.data, 1)[1]
            total += labels.size(0)

            if i == 0: print(
                f"{outputs = }\n{predicted = }\n{labels = }\n{expected = }",
                flush=True,
            )

            correct += (predicted == expected).sum().item()

        print(f"test accuracy: {100 * correct / total} %", flush=True)


if __name__ == "__main__":
    main()
