# ==== import from package ==== #
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
# ==== import from this folder ==== #
from dataset import get_dataloader

def get_latent_set(mode='VQVAE', load_epoch=49, dataset_mode='all', device=None):
    """
        This function will return a list that collect the latents generate from VAE
    """
    assert mode in ['VAE', 'VQVAE']
    if device is None:
        device = torch.device("cuda")

    eval_batch_size = 16
    net = torch.load(f'model_ckpt/{mode}/epoch_AE_{load_epoch}.pt')

    dataloader = get_dataloader(mode=dataset_mode, batch_size=eval_batch_size)
    with torch.no_grad():
        latents = []
        z_indices = []
        for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
            raw_img, seg_img, brain_idx, z_idx = [
                data.to(device) for data in batch_data]

            # Encode latent and save latents & z-index
            latent = net.encode(raw_img, z_idx)
            latents.append(latent.detach())
            z_indices.append(z_idx)

        # Concat tensor
        latents = torch.cat(latents)
        z_indices = torch.cat(z_indices)

    # Pack as dataset
    dataset = TensorDataset(latents, z_indices)
    return dataset

if __name__ == '__main__':
    mode = 'VAE'
    load_epoch = 31
    latent_set = get_latent_set(mode, load_epoch)
    torch.save(latent_set, f'collected_latents/{mode}_{load_epoch}.pt')
