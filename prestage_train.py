'''
TODO: 
1. Log in util
2. Continue Train
3. Merge eval & train
'''

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
from util import reset_dir, weight_scheduler, compact_large_image

DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)


net = Autoencoder().to(device=DEVICE)
discriminator = NLayerDiscriminator(
    input_nc=1, n_layers=3).apply(weights_init).to(device=DEVICE)

learning_rate = 5e-5
opt_ae = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.9))
opt_d = torch.optim.Adam(discriminator.parameters(),
                         lr=learning_rate, betas=(0.5, 0.9))

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

    recon_imgs = compact_large_image(recon_imgs, HZ=4, WZ=8)
    for idx, brain_idx in enumerate(brain_indices[::32]):
        plt.imsave(f'{folder}/recon_{brain_idx}.png',
                   recon_imgs[idx] * 0.5 + 0.5, cmap='gray')

    # Generate images from randn
    cur_idx = 0
    # Sample number of img_limit brains
    img_limit = 32
    # For big image format, img_limit should be multiple of 32.
    assert img_limit % 32 == 0, "img_limit should %32 == 0"
    imgs = []
    for _ in range((img_limit-1) // raw_img.shape[0] + 1):
        gen_img = net.sample(raw_img.shape[0])
        imgs.append(gen_img.detach().cpu())
        for idx, inf_img in enumerate(gen_img.detach().cpu().numpy()):
            plt.imsave(f'{folder}/gen_{cur_idx + idx}.png',
                       inf_img[0] * 0.5 + 0.5, cmap='gray')

        # Only sample img_limit images
        if cur_idx > img_limit:
            break
        cur_idx += raw_img.shape[0]

    # Gerneate one big image contain 32 brains
    imgs = torch.concat(imgs, 0)[:img_limit]
    imgs = compact_large_image(imgs, HZ=4, WZ=8)
    for idx in range(imgs.shape[0]):
        plt.imsave(f'{folder}/gen_large_{idx}.png',
                   imgs[idx] * 0.5 + 0.5, cmap='gray')


cur_iter = 0

# We define 500 iterations as an epoch.
ITER_PER_EPOCH = 500


def run_epoch(net, dataloader, update=True):
    global cur_iter
    total_num, recon_total_loss, kld_total_loss, G_total_loss, D_total_loss = 0, 0, 0, 0, 0
    total = min(ITER_PER_EPOCH, len(dataloader)) if update else len(dataloader)
    for now_step, batch_data in tqdm(enumerate(dataloader), total=total):
        raw_img, seg_img, brain_idx, z_idx = [
            data.to(DEVICE) for data in batch_data]

        # Get weight of each loss
        w_recon, w_perceptual, w_kld, w_dis = weight_scheduler(cur_iter, change_cycle=ITER_PER_EPOCH)
        # Count current iter
        cur_iter += 1

        # Train Generator
        opt_ae.zero_grad()

        recon_img, latent, kld_loss = net(raw_img)

        recon_loss = torch.abs(raw_img.contiguous() - recon_img.contiguous())
        perceptual_loss = discriminator.LPIPS(
            recon_img.contiguous(), raw_img.contiguous())
        logits_fake = discriminator(recon_img.contiguous())
        g_loss = -torch.mean(logits_fake)

        recon_loss = torch.sum(
            recon_loss + w_perceptual * perceptual_loss) / recon_loss.shape[0]
        kld_loss = torch.sum(kld_loss) / kld_loss.shape[0]

        # Adjust discriminator weight
        # If eval mode, we just set d_weight = 1 (for fast computing)
        if update:
            recon_grads = torch.autograd.grad(
                recon_loss, net.get_last_layer(), retain_graph=True)[0]
            g_grads = torch.autograd.grad(
                g_loss, net.get_last_layer(), retain_graph=True)[0]
            d_weight = torch.norm(recon_grads) / (torch.norm(g_grads) + 1e-4)
            d_weight = torch.clamp(d_weight, 0.0, 10.0).detach()
        else:
            d_weight = 1

        # 1e-4, 0.5 is hyperparameters for loss combination
        loss = w_recon * recon_loss + w_kld * kld_loss + w_dis * d_weight * g_loss

        if update:
            loss.backward()
            opt_ae.step()

        # Train Discriminator
        opt_d.zero_grad()

        # We should detach or it'll backprop generator side. (It'll occurs error that said we should retain_graph)
        recon_img = recon_img.detach()
        logits_real = discriminator(raw_img.contiguous().detach())
        logits_fake = discriminator(recon_img.contiguous().detach())
        perceptual_loss = discriminator.LPIPS(
            recon_img.contiguous(), raw_img.contiguous())

        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        loss_p = torch.mean(perceptual_loss)

        # First 0.5 is discriminator factor, Second 0.5 is from hinge loss
        d_loss = 0.5 * 0.5 * (loss_real + loss_fake) - loss_p

        if update:
            d_loss.backward()
            opt_d.step()

        recon_total_loss += recon_loss.item() * len(raw_img)
        kld_total_loss += kld_loss.item() * len(raw_img)
        G_total_loss += g_loss.item() * len(raw_img)
        D_total_loss += d_loss.item() * len(raw_img)
        total_num += len(raw_img)

        # Checkpoint
        if update and cur_iter % ITER_PER_EPOCH == 1 and cur_iter != 1:

            # Change eval mode
            net.eval()
            vis_folder = f"VAE_vis/iter_{cur_iter}"
            with torch.no_grad():
                visualize_recon(net, test_dataloader, vis_folder)
            torch.save(net, f'model_ckpt/VAE/iter_{cur_iter}.pt')

            # Change train mode
            net.train()
            return recon_total_loss / total_num, kld_total_loss / total_num, G_total_loss / total_num, D_total_loss / total_num

    return recon_total_loss / total_num, kld_total_loss / total_num, G_total_loss / total_num, D_total_loss / total_num

debug = False
if debug:
    tiny_dataloader = get_dataloader(mode='train', batch_size=6, limit=32)
    net.train()
    run_epoch(net, tiny_dataloader, update=True)


# Get dataloader
train_dataloader = get_dataloader(mode='train', batch_size=6)
valid_dataloader = get_dataloader(mode='validate', batch_size=6)
test_dataloader = get_dataloader(mode='test', batch_size=6)

for epoch in range(300):
    net.train()
    train_loss = run_epoch(net, train_dataloader, update=True)
    net.eval()
    with torch.no_grad():
        valid_loss = run_epoch(net, valid_dataloader, update=False)
    print('epoch {:>3d}: train loss: {:6.4f}/{:6.4f}/{:6.4f}/{:6.4f} valid loss: {:6.4f}/{:6.4f}/{:6.4f}/{:6.4f}'.format(
        epoch, *train_loss, *valid_loss))
