import time
import argparse

import torch
from torchvision.utils import save_image

from utils import *
from dataset import *
from modules import *

# IO Paths
GENERATED_IMG_PATH = 'predict/'
MODEL_PATH = './vqvae2.pt'         # trained model

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def inference(model: VQVAE, loader: DataLoader, device, sample_size=10):
    print("Generating...")

    for _, (image, _) in enumerate(loader): # i, (image, label)
        image = image.to(device)

        if sample_size > len(image):
            raise ValueError(f"Sample size ({sample_size}) cannot be larger than batch size ({len(image)}).")
    
        sample = image[:sample_size]

        model.eval()

        with torch.no_grad():
            out, _ = model(sample)

        save_image(
            torch.cat([sample, out], 0),
            f"{GENERATED_IMG_PATH}gen_{str(0).zfill(5)}.png",
            nrow=sample_size,
            normalize=True,
            range=(-1, 1),
        )

        break

def main(args):
    start_time = time.time()
    print("Program Starts")
    print("Device:", device)

    # Data
    print("Loading Data...")
    testloader = load_data(batch_size=args.sample_size, test=True)

    # Model
    model = VQVAE().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))   # load the given model
    
    # Test & visualize
    inference(model, testloader, device, sample_size=args.sample_size)
    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_size", type=int, default=10)

    args = parser.parse_args()

    print(args)

    main(args)
