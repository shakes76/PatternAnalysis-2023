# ==== import from package ==== #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, repeat

# ==== import from this folder ==== #
from AE_model import Autoencoder
from Discriminator import NLayerDiscriminator, weights_init
from dataset import get_dataloader
from util import reset_dir

DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)


net = Autoencoder().to(device=DEVICE)
discriminator = NLayerDiscriminator(
    input_nc=1, n_layers=3).apply(weights_init).to(device=DEVICE)

learning_rate = 4.5e-06
opt_ae = optim.Adam(net.parameters(), lr=4.5e-06, betas=(0.5, 0.9))
opt_d = torch.optim.Adam(discriminator.parameters(),
                         lr=learning_rate, betas=(0.5, 0.9))

# Get dataloader
train_dataloader = get_dataloader(mode='train', batch_size=6)
valid_dataloader = get_dataloader(mode='validate', batch_size=6)
test_dataloader = get_dataloader(mode='test', batch_size=6)

# Reset visualization folder
reset_dir('VAE_vis')
# Reset checkpoint folder
reset_dir('model_ckpt/VAE')


def visualize_recon(net, dataloader, folder):

    # Clean Directory
    reset_dir(folder)

    # Reconstruct the given data
    recon_imgs, brain_indices = [], []
    for now_step, batch_data in enumerate(dataloader):
        raw_img, seg_img, brain_idx, z_idx = [
            data.to(DEVICE) for data in batch_data]
        recon_img, latent, kld_loss = net(raw_img)
        # Record reconstructed images (for visualization) and brain indices (for labeling)
        recon_imgs.append(recon_img.detach().cpu())
        brain_indices.append(brain_idx.detach().cpu())

    recon_imgs, brain_indices = torch.concat(
        recon_imgs, 0), torch.concat(brain_indices, 0)

    # Reshape same brain but different z-index into one image
    recon_imgs = rearrange(recon_imgs, ' ( I Z ) C H W -> I Z C H W', Z=32)
    # Image should be 4 * 8 of brains
    recon_imgs = rearrange(
        recon_imgs, ' I (HZ WZ) C H W -> I (HZ H) (WZ W) C', HZ=4)
    # Repeat channel to 3 (from graysacle to RGB shape)
    recon_imgs = repeat(
        recon_imgs, 'I H W C -> I H W (repeat C)', repeat=3).numpy()

    for idx, brain_idx in enumerate(brain_indices[::32]):
        plt.imsave(f'{folder}/recon_{brain_idx}.png',
                   recon_imgs[idx] * 0.5 + 0.5, cmap='gray')

    # Generate images from randn
    cur_idx = 0
    for i in range(32):
        gen_img = net.sample(raw_img.shape[0])
        for idx, inf_img in enumerate(gen_img.detach().cpu().numpy()):
            plt.imsave(f'{folder}/gen_{cur_idx + idx}.png',
                       inf_img[0] * 0.5 + 0.5, cmap='gray')

        # Only sample 32 images
        if cur_idx > 32:
            break
        cur_idx += raw_img.shape[0]


cur_iter = 0


def run_epoch(net, dataloader, update=True):
    global cur_iter
    total_num, recon_total_loss, kld_total_loss, G_total_loss, D_total_loss = 0, 0, 0, 0, 0
    for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        raw_img, seg_img, brain_idx, z_idx = [
            data.to(DEVICE) for data in batch_data]

        # Count current iter
        cur_iter += 1

        # Train Generator
        opt_ae.zero_grad()

        recon_img, latent, kld_loss = net(raw_img)

        recon_loss = torch.abs(raw_img.contiguous() - recon_img.contiguous())
        logits_fake = discriminator(recon_img.contiguous())
        g_loss = -torch.mean(logits_fake)

        recon_loss = torch.sum(recon_loss) / recon_loss.shape[0]
        kld_loss = torch.sum(kld_loss) / kld_loss.shape[0]
        loss = recon_loss + kld_loss * 1e-6 + 0.5 * g_loss

        if update:
            loss.backward()
            opt_ae.step()

        # Train Discriminator
        opt_d.zero_grad

        logits_real = discriminator(raw_img.contiguous().detach())
        logits_fake = discriminator(recon_img.contiguous().detach())

        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        # First 0.5 is discriminator factor, Second 0.5 is from hinge loss
        d_loss = 0.5 * 0.5 * (loss_real + loss_fake)

        if update:
            d_loss.backward()
            opt_d.step()

        recon_total_loss += recon_loss.item() * len(raw_img)
        kld_total_loss += kld_loss.item() * len(raw_img)
        G_total_loss += g_loss.item() * len(raw_img)
        D_total_loss += d_loss.item() * len(raw_img)
        total_num += len(raw_img)

        # Checkpoint
        if update and cur_iter % 500 == 0:

            # Change eval mode
            net.eval()
            vis_folder = f"VAE_vis/iter_{cur_iter}"
            with torch.no_grad():
                visualize_recon(net, test_dataloader, vis_folder)
            torch.save(net, f'model_ckpt/VAE/iter_{cur_iter}.pt')

            # Change train mode
            net.train()

    return recon_total_loss / total_num, kld_total_loss / total_num, G_total_loss / total_num, D_total_loss / total_num


for epoch in range(300):
    net.train()
    train_loss = run_epoch(net, train_dataloader, update=True)
    net.eval()
    valid_loss = run_epoch(net, valid_dataloader, update=False)
    print('epoch {:>3d}: train loss: {:6.4f}/{:6.4f}/{:6.4f}/{:6.4f} valid loss: {:6.4f}/{:6.4f}/{:6.4f}/{:6.4f}'.format(
        epoch, *train_loss, *valid_loss))
