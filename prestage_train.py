# ==== import from package ==== #
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==== import from this folder ==== #
from AE_model import Autoencoder
from dataset import get_dataloader
from util import reset_dir

DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)


net = Autoencoder().cuda()

learning_rate = 4.5e-06
opt_ae = optim.Adam(net.parameters(), lr=4.5e-06, betas=(0.5, 0.9))
# opt_disc = torch.optim.Adam(DIS, lr=lr_d, betas=(0.5, 0.9))

optimizer = optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.L1Loss()

# Get dataloader
train_dataloader = get_dataloader(mode='train', batch_size=6)
valid_dataloader = get_dataloader(mode='validate', batch_size=6)
test_dataloader = get_dataloader(mode='test', batch_size=6)

# Reset visualization folder
reset_dir('VAE_vis')
# Reset checkpoint folder
reset_dir('model_ckpt/VAE')


def visualize_recon(net, dataloader, folder):
    gen_num = 32

    reset_dir(folder)
    cur_idx = 0
    for now_step, batch_data in enumerate(dataloader):
        raw_img, seg_img = batch_data
        raw_img = raw_img.to(DEVICE)
        recon_img, latent, kld_loss = net(raw_img)

        for idx, inf_img in enumerate(recon_img.detach().cpu().numpy()):
            plt.imsave(f'{folder}/recon_{cur_idx + idx}.png',
                       inf_img[0], cmap='gray')

        # Only sample gen_num images
        if cur_idx > gen_num:
            break
        cur_idx += raw_img.shape[0]

    cur_idx = 0
    for i in range(32):
        raw_img, seg_img = batch_data
        raw_img = raw_img.to(DEVICE)
        recon_img, latent, kld_loss = net(raw_img)

        for idx, inf_img in enumerate(recon_img.detach().cpu().numpy()):
            plt.imsave(f'{folder}/recon_{cur_idx + idx}.png',
                       inf_img[0], cmap='gray')

        gen_img = net.sample(raw_img.shape[0])
        for idx, inf_img in enumerate(gen_img.detach().cpu().numpy()):
            plt.imsave(f'{folder}/gen_{cur_idx + idx}.png',
                       inf_img[0], cmap='gray')

        # Only sample 32 images
        if cur_idx > 32:
            break
        cur_idx += raw_img.shape[0]


def run_epoch(net, dataloader, update=True):
    cur_iter = 0
    total_num, recon_total_loss, kld_total_loss = 0, 0, 0
    for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Count current iter
        cur_iter += 1
        optimizer.zero_grad()
        raw_img, seg_img = batch_data
        raw_img = raw_img.to(DEVICE)
        seg_img = seg_img.to(DEVICE)

        recon_img, latent, kld_loss = net(raw_img)
        recon_loss = torch.abs(raw_img.contiguous() - recon_img.contiguous())

        recon_loss = torch.sum(recon_loss) / recon_loss.shape[0]
        kld_loss = torch.sum(kld_loss) / kld_loss.shape[0]
        loss = recon_loss + kld_loss * 1e-6

        if update:
            loss.backward()
            optimizer.step()
        recon_total_loss += recon_loss.item() * len(raw_img)
        kld_total_loss += kld_loss.item() * len(raw_img)
        total_num += 1

        # Checkpoint
        if update and cur_iter % 100 == 0:

            # Change eval mode
            net.eval()
            vis_folder = f"VAE_vis/iter_{cur_iter}"
            visualize_recon(net, test_dataloader, vis_folder)
            torch.save(net, f'model_ckpt/VAE/iter_{cur_iter}.pt')

            # Change train mode
            net.train()

    return recon_total_loss / total_num, kld_total_loss / total_num


for epoch in range(300):
    net.train()
    train_loss = run_epoch(net, train_dataloader, update=True)
    net.eval()
    valid_loss = run_epoch(net, valid_dataloader, update=False)
    print('epoch {:>3d}: train loss: {:6.4f}/{:6.4f} valid loss: {:6.4f}/{:6.4f}'.format(
        epoch, *train_loss, *valid_loss))
