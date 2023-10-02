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
from einops import rearrange, repeat, reduce
from collections import defaultdict
# ==== import from this folder ==== #
from model_VQVAE import VQVAE
from discriminator import NLayerDiscriminator, weights_init
from dataset import get_dataloader
from util import reset_dir, weight_scheduler, compact_large_image
from logger import Logger
from SSIM import ssim
DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)

net = VQVAE().to(device=DEVICE)
discriminator = NLayerDiscriminator(
    input_nc=1, n_layers=3).apply(weights_init).to(device=DEVICE)

learning_rate = 2e-4
opt_vqvae = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.5, 0.9))
opt_d = torch.optim.Adam(discriminator.parameters(),
                         lr=learning_rate, betas=(0.5, 0.9))


vis_folder = 'VQVAE_vis'
ckpt_folder = 'model_ckpt/VQVAE'

# Keep training if epoch is not zero
start_epoch = 51
if start_epoch != 0:
    # For example, if we start at epoch 7 and we need to load epoch 6.
    try:
        net = torch.load(f'{ckpt_folder}/epoch_AE_{start_epoch-1}.pt')
        discriminator = torch.load(f'{ckpt_folder}/epoch_D_{start_epoch-1}.pt')
    except Exception as e:
        print("Fail to load model")
else:
    # Reset visualization folder
    reset_dir(vis_folder)
    # Reset checkpoint folder
    reset_dir(ckpt_folder)

# logger for record losses
logger = Logger(file_name='VQVAE_log.txt', reset=(start_epoch == 0))

# We define 500 iterations as an epoch.
ITER_PER_EPOCH = 500
batch_size = 6

# Stage threshold
disc_start_iter = 250
auxiliary_start_epoch = 0

cur_iter = ITER_PER_EPOCH * start_epoch


def calculate_weight_sampler(net, dataloader):
    indices = []
    for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
        raw_img, seg_img, brain_idx, z_idx = [
            data.to(DEVICE) for data in batch_data]
        with torch.no_grad():
            batch_size = raw_img.shape[0]
            latent = net.encode(raw_img)
            quant, diff_loss, (_, _, ind) = net.quantize(latent)
            ind = rearrange(ind, '(b c h w) -> b c h w', b=batch_size,
                            h=net.z_shape[0], w=net.z_shape[1])
            indices.append(ind.detach())
    indices = torch.cat(indices)

    indices = reduce(indices, 'b c h w -> b h w', 'min')
    indices = rearrange(indices, 'b h w -> h w b')
    weight_sampler = torch.tensor(([
        [list(torch.bincount(indices[i, j], minlength=net.n_embed+1)) for i in range(net.z_shape[0])] for j in range(net.z_shape[1])
    ]))
    net.update_sampler(weight_sampler)


def train_epoch(net, dataloader, auxiliary=True):
    global cur_iter
    epoch_info = defaultdict(lambda: 0)
    for now_step, batch_data in tqdm(enumerate(dataloader), total=min(ITER_PER_EPOCH, len(dataloader))):
        raw_img, seg_img, brain_idx, z_idx = [
            data.to(DEVICE) for data in batch_data]

        # Get weight of each loss
        w_recon, w_perceptual, w_kld, w_dis = weight_scheduler(
            cur_iter, change_cycle=ITER_PER_EPOCH)

        # Train Generator
        opt_vqvae.zero_grad()

        recon_img, diff_loss, ind = net(raw_img)

        # Reconstruction Term (L1 Loss)
        recon_loss = torch.abs(raw_img.contiguous() - recon_img.contiguous())
        recon_loss = torch.sum(recon_loss) / recon_loss.shape[0]

        # Reconstruction Term (GAN Loss)
        logits_fake = discriminator(recon_img.contiguous())
        g1_loss = -torch.mean(logits_fake)

        # Adjust discriminator weight
        recon_grads = torch.norm(torch.autograd.grad(
            recon_loss, net.get_decoder_last_layer(), retain_graph=True)[0]).detach()
        g1_grads = torch.norm(torch.autograd.grad(
            g1_loss, net.get_decoder_last_layer(), retain_graph=True)[0]).detach()
        d1_weight = recon_grads / (g1_grads + 1e-4)
        d1_weight = torch.clamp(d1_weight, 0.0, 1e4).detach()

        # Apply auxiliary loss (gen from sample and trained as GAN)
        if auxiliary:
            net.eval()
            gen_img = net.sample(raw_img.shape[0])
            logits_gen = discriminator(gen_img.contiguous())
            g2_loss = -torch.mean(logits_gen)
            g2_grads = torch.norm(torch.autograd.grad(
                g2_loss, net.get_decoder_last_layer(), retain_graph=True)[0]).detach()
            d2_weight = recon_grads / (g2_grads + 1e-4)
            d2_weight = torch.clamp(d2_weight, 0.0, 1e4).detach()

        loss = w_recon * recon_loss + 1.0 * diff_loss.mean() + w_dis * \
            d1_weight * g1_loss
        if auxiliary:
            loss = loss + w_dis * d2_weight * g2_loss

        loss.backward()
        opt_vqvae.step()

        # Train Discriminator
        opt_d.zero_grad()

        # We should detach or it'll backprop generator side. (It'll occurs error that said we should retain_graph)
        recon_img = recon_img.detach()
        logits_real = discriminator(raw_img.contiguous().detach())
        logits_fake = discriminator(recon_img.contiguous().detach())
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))

        if auxiliary:
            logits_gen = discriminator(gen_img.contiguous().detach())
            loss_gen = torch.mean(F.relu(1. + logits_gen))
            # First 0.5 is discriminator factor, Second 0.5 is from hinge loss
            d_loss = 0.5 * 0.5 * (loss_real + loss_fake + loss_gen)
        else:
            d_loss = 0.5 * 0.5 * (loss_real + loss_fake)

        d_loss.backward()
        opt_d.step()

        # info to dict
        cur_info = {
            'recon_loss': recon_loss.item(),
            'diff_loss': diff_loss.item(),
            'fake_recon_loss': g1_loss.item(),
            'discriminator_loss': d_loss.item(),
            'w_recon': w_recon,
            'w_perceptual': w_perceptual,
            "w_dis": w_dis * d1_weight,
        }
        if auxiliary:
            cur_info.update({
                'fake_sample_loss': g2_loss.item(),
                "w_sample": w_dis * d2_weight,
            })

        # record epoch info
        for k, v in cur_info.items():
            epoch_info[k] += v * len(raw_img)
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
    total_ssim = 0
    recon_imgs, brain_indices = [], []
    for now_step, batch_data in enumerate(dataloader):
        raw_img, seg_img, brain_idx, z_idx = [
            data.to(DEVICE) for data in batch_data]
        recon_img, diff, info = net(raw_img)
        # Record reconstructed images (for visualization) and brain indices (for labeling)
        recon_imgs.append(recon_img.detach().cpu())
        brain_indices.append(brain_idx.detach().cpu())
        total_ssim += ssim(raw_img* 0.5 + 0.5, recon_img * 0.5 + 0.5).item() * raw_img.shape[0]

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

    return total_ssim / len(dataloader.dataset)
# Get dataloader
train_dataloader = get_dataloader(
    mode='train_and_validate', batch_size=batch_size)
test_dataloader = get_dataloader(mode='test', batch_size=16)

start_auxiliary = False
for epoch in range(start_epoch, 50):
    if not start_auxiliary and epoch >= auxiliary_start_epoch:
        print(
            f"To adapt auxiliary, we shrink the batch size from {batch_size} -> {batch_size // 2}")
        train_dataloader = get_dataloader(
            mode='train_and_validate', batch_size=batch_size // 2)
        start_auxiliary = True

    # The format string parse epoch info
    def fmt(epoch_info):
        ks = ['recon_loss', 'diff_loss', 'fake_recon_loss',
              'fake_sample_loss', 'discriminator_loss']
        return ' '.join(f"{k[:-5]}: {epoch_info[k]:6.4f}" for k in ks if k in epoch_info)
    net.train()
    train_info = train_epoch(net, train_dataloader, auxiliary=start_auxiliary)
    net.eval()
    with torch.no_grad():
        calculate_weight_sampler(net, train_dataloader)
        ssim_score = test_epoch(net, test_dataloader, f'{vis_folder}/epoch_{epoch}')

    # Save the model
    torch.save(net, f'{ckpt_folder}/epoch_AE_{epoch}.pt')
    torch.save(discriminator, f'{ckpt_folder}/epoch_D_{epoch}.pt')

    print('{:=^110s}'.format(f' epoch {epoch:>3d} '))
    print(fmt(train_info), f' Test SSIM: {ssim_score:2.4f}')
