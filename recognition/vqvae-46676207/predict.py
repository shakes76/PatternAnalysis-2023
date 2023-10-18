""" Predicting module for VQVAE2 """

import time
import argparse

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import numpy as np

from dataset import load_data
from modules import VQVAE

# IO Paths
GENERATED_IMG_PATH = 'predict/'     # path to save the generated images
MODEL_PATH = './vqvae2.pt'          # trained model

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu when cuda is available

def inference(model: VQVAE, loader: DataLoader, device, sample_size=10, verbose=False):
    """
    Test the given VQVAE model: visualize the output and print SSIM
    
    Args:
        model: the trained VQVAE model for image generating
        loader: a dataloader of testing dataset
        device: the device (cpu/gpu) to use for inference
        sample_size: number of images to generate
        verbose: print details for each image or not
    """
    print("Generating...")

    for _, (image, _) in enumerate(loader): # i, (image, label)
        image = image.to(device)

        if sample_size > len(image):
            raise ValueError(f"Sample size ({sample_size}) cannot be larger than batch size ({len(image)}).")
    
        sample = image[:sample_size]

        model.eval()                                # set the model in evaluation mode

        with torch.no_grad():                       # disable gradient calculation
            out, _ = model(sample)                  # get generated imgs

        save_image(
            torch.cat([sample, out], 0),            # concatenates sequence in dimension 0
            f"{GENERATED_IMG_PATH}gen_{str(0).zfill(5)}.png",   # file name
            nrow=sample_size,                       # number of samples
            normalize=True,                         # normalize
            range=(-1, 1),                          # range of data
        )

        # SSIM
        ssim_values = []                            # initialize the list
        out = np.squeeze(out.cpu().numpy())         # convert to numpy array
        sample = np.squeeze(sample.cpu().numpy())   # convert to numpy array
        for o, s in zip(out, sample):
            ssim_values.append(ssim(o, s, data_range=np.ptp(s,axis=(0,1))))         # get ssim
            if verbose: print("Range out: %.4f  Range sample: %s  SSIM: %.4f" %     # print info of each img
                            (np.ptp(o,axis=(0,1)), np.ptp(s,axis=(0,1)), ssim_values[-1]))  
        print(f"SSIM: {sum(ssim_values)/sample_size}") # print average ssim

        break

def main(args):
    start_time = time.time()    # tracking execution time
    print("Program Starts")
    print("Device:", device)

    # Data
    print("Loading Data...")
    testloader = load_data(batch_size=args.sample_size, test=True)  # get dataloader

    # Model
    model = VQVAE().to(device)                      # send to gpu if available
    model.load_state_dict(torch.load(MODEL_PATH))   # load the given model
    
    # Test & visualize
    inference(model, testloader, device, sample_size=args.sample_size, verbose=args.verbose)    # inference start
    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))   # print execution time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_size", type=int, default=10)  # how many sample imgs do you want
    parser.add_argument("--verbose", type=bool, default=False)  # print info of each img

    args = parser.parse_args()
    print(args)

    main(args)
