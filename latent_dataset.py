# ==== import from package ==== #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
# ==== import from this folder ==== #
from dataset import get_dataloader
from logger import Logger


def collate_fn(net, batch, mode):
    latents, z_index = zip(*batch)
    if mode == 'VAE':
        latents = net.reparameterization(latents)
    return latents, z_index

def get_latent_set(mode='VQVAE', load_epoch=49, dataset_mode='all', device=None):
    assert mode in ['VAE', 'VQVAE']
    if device is None:
        device = torch.device("cuda")

    eval_batch_size = 16
    net = torch.load(f'model_ckpt/{mode}/epoch_AE_{load_epoch}.pt')
    print(dataset_mode)
    dataloader = get_dataloader(mode=dataset_mode, batch_size=eval_batch_size)
    with torch.no_grad():
        latents = []
        z_indices = []
        for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            raw_img, seg_img, brain_idx, z_idx = [
                data.to(device) for data in batch_data]
            batch_size = raw_img.shape[0]
            latent = net.encode(raw_img, z_idx)
            latent, diff_loss, ind = net.quantize(latent)
            latents.append(latent.detach())
            z_indices.append(z_idx)
        latents = torch.cat(latents)
        z_indices = torch.cat(z_indices)
    dataset = TensorDataset(latents, z_indices)
    return dataset

if __name__ == '__main__':
    mode = 'VQVAE'
    load_epoch = 45
    test_save = True
    if test_save:
        latent_set = get_latent_set(mode, load_epoch)
        torch.save(latent_set, f'collected_latents/{mode}_{load_epoch}.pt')
    else:
        latent_set = torch.load(f'collected_latents/{mode}_{load_epoch}.pt')

        dataloader = DataLoader(dataset=latent_set, batch_size=16, shuffle=True)

        for latent, z_index in dataloader:
            print(latent.shape, z_index.shape)
            break

