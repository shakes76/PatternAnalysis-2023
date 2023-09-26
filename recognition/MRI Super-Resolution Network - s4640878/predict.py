import sys
import torch, torchvision
from matplotlib import pyplot as plt

from dataset import Dataset, machine
from modules import Model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    """ model """
    model = Model().to(device)
    print(f"params: {sum([p.nelement() for p in model.parameters()])}", flush=True)

    """ load datasets """
    test_loader = Dataset(train=False).loader()

    with open(file="models/sr_model.pt", mode="rb") as f:
        model.load_state_dict(
            state_dict=torch.load(f=f, map_location=torch.device('cpu')),
        )

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        downsampled = torchvision.transforms.Resize(60, antialias=True)(images).to(device)

        if i == 0:

            with torch.no_grad():
                outputs = model(downsampled)

            plt.subplot(1, 3, 1)
            plt.imshow(
                (
                    (images[i] - torch.min(images[i])) / (torch.max(images[i]) - torch.min(images[i]))
                ).permute(1, 2, 0).cpu()
            )
            plt.title("original", size=8)

            plt.subplot(1, 3, 2)
            plt.imshow(
                (
                    (downsampled[i] - torch.min(downsampled[i])) / (torch.max(downsampled[i]) - torch.min(downsampled[i]))
                ).permute(1, 2, 0).cpu()
            )
            plt.title("downsampled", size=8)

            plt.subplot(1, 3, 3)
            plt.imshow(
                (
                    (outputs[i] - torch.min(outputs[i])) / (torch.max(outputs[i]) - torch.min(outputs[i]))
                ).permute(1, 2, 0).cpu()
            )
            plt.title("upsampled", size=8)

            # print(f"{images[i].shape = }, {downsampled[i].shape = }")  # images[i].shape = torch.Size([3, 240, 256]), downsampled.shape = torch.Size([3, 60, 64])

            plt.tight_layout()

            if machine == "local":
                plt.savefig("./debug/predict.png")
            elif machine == "rangpur":
                plt.savefig("./outputs/predict.png")

            sys.exit()



if __name__ == "__main__":
    main()