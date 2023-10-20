""" Predicting module for VQVAE2 """

import os
import time
import argparse

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm

from dataset import load_data
from modules import VQVAE, PixelSNAIL

# IO Paths
GENERATED_IMG_PATH = 'predict/'                 # path to save the generated images
MODEL_VQVAE_PATH = './vqvae2.pt'                # trained VQVAE model
MODEL_TOP_PATH = './pixelsnail_top.pt'          # trained PixelSNAIL model (top)
MODEL_BOTTOM_PATH = './pixelsnail_bottom.pt'    # trained PixelSNAIL model (bottom)

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use gpu when cuda is available

def inference(model: VQVAE, 
              loader: DataLoader, 
              device, 
              sample_size=10, 
              random=False, 
              verbose=False):
    """
    Test the given VQVAE model: visualize the output and print SSIM
    
    Args:
        model: the trained VQVAE model for image generating
        loader: a dataloader of testing dataset
        device: the device (cpu/gpu) to use for inference
        sample_size: number of images to generate
        random: generate random new image or simply do reconstruction
        verbose: print details for each image or not
    """
    print("Generating...")

    for i, (image, _) in enumerate(loader): # i, (image, label)
        image = image.to(device)

        if sample_size > len(image):
            raise ValueError(f"Sample size ({sample_size}) cannot be larger than batch size ({len(image)}).")
    
        sample = image[:sample_size]

        model.eval()                                # set the model in evaluation mode

        with torch.no_grad():                       # disable gradient calculation
            if random:
                quant_t, quant_b, _, _, _ = model.encode(sample)
                quant_t += 0.05 * torch.randn_like(quant_t)
                quant_b += 0.05 * torch.randn_like(quant_b)
                out = model.decode(quant_t, quant_b)
            else:
                out, _ = model(sample)                  # get reconstructed imgs

        save_image(
            torch.cat([sample, out], 0),            # concatenates sequence in dimension 0
            f"{GENERATED_IMG_PATH}rec_{str(i).zfill(5)}.png",   # file name
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

@torch.no_grad()
def sample_model(model, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row

def load_model(model, checkpoint):
    ckpt = torch.load(checkpoint)
    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

    elif model == 'top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif model == 'bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )
        
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model

def main(args):
    start_time = time.time()    # tracking execution time
    print("Program Starts")
    print("Device:", device)

    # Model
    if args.generate:
        model_vqvae = load_model('vqvae', MODEL_VQVAE_PATH)
        model_top = load_model('top', MODEL_TOP_PATH)
        model_bottom = load_model('bottom', MODEL_BOTTOM_PATH)
        
        top_sample = sample_model(model_top, args.batch, [32, 32], args.temp)
        bottom_sample = sample_model(model_bottom, args.batch, [64, 64], args.temp, condition=top_sample)
        decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
        decoded_sample = decoded_sample.clamp(-1, 1)

        save_image(decoded_sample, f"{GENERATED_IMG_PATH}gen.png", normalize=True, range=(-1, 1))

    else:
        # Data
        print("Loading Data...")
        testloader = load_data(batch_size=args.sample_size, test=True)  # get dataloader
        # Model
        model = VQVAE().to(device)                      # send to gpu if available
        model.load_state_dict(torch.load(MODEL_VQVAE_PATH))   # load the given model
        # Reconstruct
        inference(model, testloader, device, sample_size=args.sample_size, verbose=args.verbose)    # inference start

    # Test & visualize
    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))   # print execution time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_size", type=int, default=10)  # how many sample imgs do you want
    parser.add_argument("--generate", type=bool, default=True)  # Generate or reconstructure
    parser.add_argument("--verbose", type=bool, default=False)  # print info of each img
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--temp', type=float, default=1.0)

    args = parser.parse_args()
    print(args)

    main(args)
