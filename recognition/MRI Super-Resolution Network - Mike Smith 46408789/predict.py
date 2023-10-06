import sys
import torch, torchvision
from matplotlib import pyplot as plt

from dataset import Dataset, machine
from modules import Model_Generator


def main():

    try: n = int(sys.argv[1]); n = n if n < 5 else 5
    except Exception: n = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pytorch version: {torch.__version__}, exe device: {device}", flush=True)

    """ model """
    model = Model_Generator().to(device)
    print(f"params: {sum([p.nelement() for p in model.parameters()])}", flush=True)

    """ load testing dataset """
    test_loader = Dataset(train=False).loader()

    with open(file="models/sr_model_gan_2.pt", mode="rb") as f:
        model.load_state_dict(
            state_dict=torch.load(f=f, map_location=torch.device('cpu')),
        )

    plt.style.use("dark_background")
    
    model.eval()
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        downsampled = torchvision.transforms.Resize(60, antialias=True)(images).to(device)

        with torch.no_grad():
            outputs = model(downsampled)

        plt.subplot(n, 3, i*3 + 1)
        plt.imshow(
            (
                (images[0] - torch.min(images[0])) / (torch.max(images[0]) - torch.min(images[0]))
            ).permute(1, 2, 0).cpu()
        )
        plt.title("Original", size=10)

        plt.subplot(n, 3, i*3 + 2)
        plt.imshow(
            (
                (downsampled[0] - torch.min(downsampled[0])) / (torch.max(downsampled[0]) - torch.min(downsampled[0]))
            ).permute(1, 2, 0).cpu()
        )
        plt.title("Downsampled", size=10)

        plt.subplot(n, 3, i*3 + 3)
        plt.imshow(
            (
                (outputs[0] - torch.min(outputs[0])) / (torch.max(outputs[0]) - torch.min(outputs[0]))
            ).permute(1, 2, 0).cpu()
        )
        plt.title("Reconstructed", size=10)

        if i == n - 1: break

    plt.tight_layout()
    if machine == "local": 
        plt.show()
    elif machine == "rangpur":
        plt.savefig("./outputs/predict.png")


if __name__ == "__main__":
    main()