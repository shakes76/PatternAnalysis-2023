""" Predicting module for VQVAE2 """

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
        model: The trained VQVAE model for image generating
        loader: A dataloader of testing dataset
        device: The device (cpu/gpu) to use for inference
        sample_size: Number of images to generate
        random: Generate random new image or simply do reconstruction
        verbose: Print details for each image or not
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
            f"{GENERATED_IMG_PATH}rec_{str(sample_size).zfill(5)}.png",   # file name
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


@torch.no_grad() # do not compute gradients
def sample_model(model: PixelSNAIL, batch, size, temperature, condition=None):
    """
    Returns a sample output from the given PixelSNAIL model

    Args:
        model: A PixelSNAIL model for generating samples
        batch: Batch size
        size: Shape of input sample
        temperature: Adjust the softmax probability distribution (higher means more uncertainty)
        condition: Optional tensor for conditional generation

    Returns:
        row: Generated samples
    """
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)   # initialize with zeros
    cache = {}                                                      # store model states

    for i in tqdm(range(size[0])):                                  # loop over the hight
        for j in range(size[1]):                                    # loop over the width
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)    # get output & model state
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)                      # get probability
            sample = torch.multinomial(prob, 1).squeeze(-1)                             # draw a sample
            row[:, i, j] = sample                                                       # update with the sample

    return row                                                      # return generated samples


def load_model(model_name, checkpoint):
    """
    Load and return the specified model

    Args:
        model_name: Name of the model to load (vqvae/top/bottom)
        checkpoint: Path of a trained model with its args

    Returns:
        model: A loaded model
    """
    ckpt = torch.load(checkpoint)   # load the given model with its args
    
    if 'args' in ckpt:              # get the model args
        args = ckpt['args']

    match model_name:
        case 'vqvae':               # initialize VQVAE model
            model = VQVAE()

        case 'top':                 # initialize PixelSNAIL model (top)
            model = PixelSNAIL(     # see details of args in modules.py
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

        case 'bottom':              # initialize PixelSNAIL model (bottom)
            model = PixelSNAIL(     # see details of args in modules.py
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
        
    if 'model' in ckpt:             # get the trained model
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)     # load the trained model
    model = model.to(device)        # use gpu if available
    model.eval()                    # set to evaluation mode

    return model                    # return the trained model


def main(args):
    """
    The main entry point of prediction

    Args:
        args: All the arguments from shell, listed at the end of the file
    """
    start_time = time.time()        # tracking execution time
    print("Program Starts")         # show the process is begin
    print("Device:", device)        # print the using device

    # Model
    if args.generate:               # generate new images
        model_vqvae = load_model('vqvae', MODEL_VQVAE_PATH)     # load a trained VQVAE model
        model_top = load_model('top', MODEL_TOP_PATH)           # load a trained PixelSNAIL model (top)
        model_bottom = load_model('bottom', MODEL_BOTTOM_PATH)  # load a trained PixelSNAIL model (bottom)
        
        top_sample = sample_model(model_top, args.sample_size, [32, 32], args.temp)                               # get the code of top-hier
        bottom_sample = sample_model(model_bottom, args.sample_size, [64, 64], args.temp, condition=top_sample)   # get the code of bottom-hier
        decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)                                 # converte code to image
        decoded_sample = decoded_sample.clamp(-1, 1)                                                        # clamps to valid pixel values

        save_image(decoded_sample, f"{GENERATED_IMG_PATH}gen_{str(args.sample_size).zfill(5)}.png", normalize=True, range=(-1, 1))           # save image

    else:                           # encode and reconstruct images
        # Data
        print("Loading Data...")
        testloader = load_data(batch_size=args.sample_size, test=True)                              # get dataloader
        # Model
        model = VQVAE().to(device)                                                                  # initialize the model & send to gpu if available
        model.load_state_dict(torch.load(MODEL_VQVAE_PATH))                                         # load the given model
        # Reconstruct
        inference(model, testloader, device, sample_size=args.sample_size, verbose=args.verbose)    # save reconstructed imgs and print SSIM

    # Test & visualize
    print("Execution Time: %.2f min" % ((time.time() - start_time) / 60))                           # print execution time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sample_size", type=int, default=5)  # number of sample imgs to show
    parser.add_argument("--generate", type=bool, default=True)  # Generate or reconstructure
    parser.add_argument("--verbose", type=bool, default=False)  # print info of each img
    parser.add_argument('--temp', type=float, default=1.0)      # temperature

    args = parser.parse_args()
    print(args)

    main(args)
