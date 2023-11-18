import sys
import torch, torchvision
from matplotlib import pyplot as plt

from dataset import Dataset, machine
from modules import Model_Generator


def main():
    """ 
    specify the number of predictions to be made 
    usage: py predict.py [n | default = 3]
    
    """
    try: n = int(sys.argv[1]); n = n if n < 5 else 5
    except Exception: n = 3

    """ get the pytorch exe device and pytorch version """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pytorch version: {torch.__version__}, exe device: {device}", flush=True)

    """ load in the generator model and output the total number of params"""
    model = Model_Generator().to(device)
    print(f"params: {sum([p.nelement() for p in model.parameters()])}", flush=True)

    """ load testing dataset """
    test_loader = Dataset(train=False).loader()

    """ open the model state dict file """
    with open(file="models/sr_model_gan.pt", mode="rb") as f:
        model.load_state_dict(
            state_dict=torch.load(f=f, map_location=torch.device('cpu')),
        )

    """ set dark bg for plots """
    plt.style.use("dark_background")
    
    """ evaluate the model """
    model.eval()
    for i, (images, labels) in enumerate(test_loader):

        """ send images and labels to hardware """
        images = images.to(device)
        labels = labels.to(device)

        """ downsample the images (240 x 256 -> 60 x 64) """
        downsampled = torchvision.transforms.Resize(60, antialias=True)(images).to(device)

        """ make the predictions """
        with torch.no_grad():
            outputs = model(downsampled)

        """ 
        plot the outputs

        """
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

    """ save or show the output results """
    plt.tight_layout()
    if machine == "local": 
        plt.show()
    elif machine == "rangpur":
        plt.savefig("./outputs/predict.png")


if __name__ == "__main__":
    main()