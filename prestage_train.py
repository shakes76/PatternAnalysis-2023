import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from AE_model import Autoencoder
from dataset import get_dataloader

DEVICE = torch.device("cuda")
print("DEVICE:", DEVICE)


net = Autoencoder().cuda()

learning_rate = 4.5e-06
opt_ae = optim.Adam(net.parameters(), lr= 4.5e-06, betas=(0.5, 0.9))
# opt_disc = torch.optim.Adam(DIS, lr=lr_d, betas=(0.5, 0.9))

optimizer = optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.L1Loss()

def run_epoch(net, dataloader, update=True):
    total_num, recon_total_loss, kld_total_loss = 0, 0, 0
    for now_step, batch_data in tqdm(enumerate(dataloader), total=len(dataloader)):
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

    return recon_total_loss / total_num, kld_total_loss / total_num


train_dataloader = get_dataloader(mode='train', batch_size=6)
valid_dataloader = get_dataloader(mode='validate', batch_size=6)
test_dataloader = get_dataloader(mode='test', batch_size=6)

for epoch in range(300):
    net.train()
    train_loss = run_epoch(net, train_dataloader, update=True)
    net.eval()
    valid_loss = run_epoch(net, valid_dataloader, update=False)
    print('epoch {:>3d}: train loss: {:6.4f}/{:6.4f} valid loss: {:6.4f}/{:6.4f}'.format(
        epoch, *train_loss, *valid_loss))

    # Visualize
    import shutil
    import os
    try :
        os.mkdir(f'VAE_vis')
        print(f"Create VAE_vis dir at first time.")
    except Exception as e:
        try :
            shutil.rmtree(f'VAE_vis/epoch_{epoch}')
        except Exception as e:
            print(f"Create VAE_vis/epoch_{epoch} dir at first time.")


    os.mkdir(f'VAE_vis/epoch_{epoch}')

    for now_step, batch_data in enumerate(test_dataloader):
        raw_img, seg_img = batch_data
        raw_img = raw_img.to(DEVICE)
        recon_img, latent, kld_loss = net(raw_img)
        break
    for idx, inf_img in enumerate(recon_img.detach().cpu().numpy()):
        plt.imsave(f'VAE_vis/epoch_{epoch}/{idx}.png', inf_img[0], cmap='gray')
    # Save
    torch.save(net, 'VAE.pt')
