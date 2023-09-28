'''
TODO: 
1. Log Visualization
2. Deploy VQ-VAE
?. Time encoding for VAE
'''

# ==== import from package ==== #
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange, repeat
from collections import defaultdict
# ==== import from this folder ==== #
from AE_model import Autoencoder
from discriminator import NLayerDiscriminator, weights_init
from dataset import get_dataloader
from util import reset_dir, weight_scheduler, compact_large_image
from logger import Logger
DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)

# logger for record losses
logger = Logger(file_name='VAE_log.txt', reset=True)

net = Autoencoder().to(device=DEVICE)
discriminator = NLayerDiscriminator(
    input_nc=1, n_layers=3).apply(weights_init).to(device=DEVICE)

learning_rate = 2e-4
opt_ae = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.9))
opt_d = torch.optim.Adam(discriminator.parameters(),
                         lr=learning_rate, betas=(0.5, 0.9))

# Reset visualization folder
reset_dir('VAE_vis')
# Reset checkpoint folder
reset_dir('model_ckpt/VAE')


cur_iter = 0

# We define 500 iterations as an epoch.
ITER_PER_EPOCH = 500


def train_epoch(net, dataloader):
    global cur_iter
    epoch_info = defaultdict(lambda: 0)
    for now_step, batch_data in tqdm(enumerate(dataloader), total=min(ITER_PER_EPOCH, len(dataloader))):
        raw_img, seg_img, brain_idx, z_idx = [
            data.to(DEVICE) for data in batch_data]

        # Get weight of each loss
        w_recon, w_perceptual, w_kld, w_dis = weight_scheduler(
            cur_iter, change_cycle=ITER_PER_EPOCH)

        # Train Generator
        opt_ae.zero_grad()

        recon_img, latent, kld_loss = net(raw_img)
        gen_img = net.sample(raw_img.shape[0])

        recon_loss = torch.abs(raw_img.contiguous() - recon_img.contiguous())
        perceptual_loss = discriminator.LPIPS(
            recon_img.contiguous(), raw_img.contiguous())
        logits_fake = discriminator(recon_img.contiguous())
        logits_gen = discriminator(gen_img.contiguous())
        g1_loss = -torch.mean(logits_fake)
        g2_loss = -torch.mean(logits_gen)

        recon_loss = torch.sum(
            recon_loss + w_dis * w_perceptual * perceptual_loss) / recon_loss.shape[0]
        kld_loss = torch.sum(kld_loss) / kld_loss.shape[0]

        # Adjust discriminator weight
        recon_grads = torch.norm(torch.autograd.grad(
            recon_loss, net.get_decoder_last_layer(), retain_graph=True)[0]).detach()
        g1_grads = torch.norm(torch.autograd.grad(
            g1_loss, net.get_decoder_last_layer(), retain_graph=True)[0]).detach()
        g2_grads = torch.norm(torch.autograd.grad(
            g2_loss, net.get_decoder_last_layer(), retain_graph=True)[0]).detach()
        kl_grads = torch.norm(torch.autograd.grad(
            kld_loss, net.get_encoder_last_layer(), retain_graph=True)[0]).detach()

        kl_weight = 0.01 * recon_grads / (kl_grads + 1e-4)
        d1_weight = 0.01 * recon_grads / (g1_grads + 1e-4)
        d2_weight = 0.01 * recon_grads / (g2_grads + 1e-4)
        d1_weight = torch.clamp(d1_weight, 0.0, 1e4).detach()
        d2_weight = torch.clamp(d2_weight, 0.0, 1e4).detach()
        
        loss = w_recon * recon_loss + kl_weight * w_kld * kld_loss +  \
            w_dis * d1_weight * g1_loss + \
            w_dis * d2_weight * g2_loss
        
        loss.backward()
        opt_ae.step()

        # Train Discriminator
        opt_d.zero_grad()

        # We should detach or it'll backprop generator side. (It'll occurs error that said we should retain_graph)
        recon_img = recon_img.detach()
        logits_real = discriminator(raw_img.contiguous().detach())
        logits_fake = discriminator(recon_img.contiguous().detach())
        logits_gen = discriminator(gen_img.contiguous().detach())
        perceptual_loss = discriminator.LPIPS(
            recon_img.contiguous(), raw_img.contiguous())

        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        loss_gen = torch.mean(F.relu(1. + logits_gen))
        loss_p = torch.mean(perceptual_loss)

        # First 0.5 is discriminator factor, Second 0.5 is from hinge loss
        d_loss = 0.5 * 0.5 * (loss_real + loss_fake + loss_gen) - loss_p
        # d_loss = 0.5 * 0.5 * (loss_real + loss_fake ) - loss_p

        d_loss.backward()
        opt_d.step()

        # info to dict
        cur_info = {
            'recon_loss' : recon_loss.item(),
            'kld_loss' : kld_loss.item(),
            'fake_recon_loss' : g1_loss.item(),
            'fake_sample_loss' : g2_loss.item(),
            'discriminator_loss' : d_loss.item(),
            'w_recon': w_recon,
            'w_perceptual': w_perceptual,
            'w_kld': w_kld * kl_weight,
            "w_recon" : w_dis * d1_weight,
            "w_sample" : w_dis * d2_weight,
        }

        # record epoch info
        for k, v in cur_info.items():
            epoch_info[k] += v
        epoch_info['total_num'] += len(raw_img)

        # Only if update should count cur_iter and count log
        logger.update_dict(cur_info)
        cur_iter += 1

        if now_step % ITER_PER_EPOCH == 0 and now_step != 0:
            break
    # Mean the info
    for k in epoch_info:
        if k != 'total_num':
            epoch_info[k] /= epoch_info['total_num']

    return epoch_info


def test_epoch(net, dataloader, folder):

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


batch_size = 3
debug = False
if debug:
    tiny_dataloader = get_dataloader(
        mode='train', batch_size=batch_size, limit=32)
    net.train()
    train_epoch(net, tiny_dataloader, update=True)


# Get dataloader
train_dataloader = get_dataloader(mode='train_and_validate', batch_size=batch_size)
test_dataloader = get_dataloader(mode='test', batch_size=batch_size)

for epoch in range(300):
    # The format string parse epoch info
    def fmt(epoch_info):
        ks = ['recon_loss', 'kld_loss', 'fake_recon_loss',
              'fake_sample_loss', 'discriminator_loss']
        return ' '.join(f"{k[:-5]}: {epoch_info[k]:6.4f}" for k in ks if k in epoch_info)

    net.train()
    train_info = train_epoch(net, train_dataloader)
    net.eval()
    with torch.no_grad():
        test_epoch(net, test_dataloader, f'VAE_vis/epoch_{epoch}')
    # Save the model
    torch.save(net, f'model_ckpt/VAE/epoch_{epoch}.pt')

    print('{:=^100s}'.format(f' epoch {epoch:>3d} '))
    print(fmt(train_info))