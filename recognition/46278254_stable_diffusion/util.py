import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import math

from einops import rearrange, repeat

def reset_dir(folder):
    """
        reset folder: remove all the files in this folder. If this folder is not exist, then it'll make new one.
    """

    # Try to remove folder
    try:
        shutil.rmtree(folder)
    except Exception as e:
        # This folder is not exist.
        pass
    os.mkdir(folder)


def compact_large_image(imgs, HZ=4, WZ=8):
    # Reshape same brain but different z-index into one image
    imgs = rearrange(imgs, ' ( I Z ) C H W -> I Z C H W', Z=HZ*WZ)
    # Image should be 4 * 8 of brains
    imgs = rearrange(
        imgs, ' I (HZ WZ) C H W -> I (HZ H) (WZ W) C', HZ=HZ)
    # Repeat channel to 3 (from graysacle to RGB shape)
    imgs = repeat(
        imgs, 'I H W C -> I H W (repeat C)', repeat=3).numpy()
    return imgs


def weight_scheduler(cur_iter=0, end=50000, change_cycle=500, disc_start=10000):
    cur_iter = min(cur_iter, end)
    w_recon = 1

    # https://github.com/haofuml/cyclical_annealing
    # Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing (NAACL 2019):
    w_kld = 1
    # first half is linear, end half is constant
    iter_in_cycle = cur_iter % change_cycle
    if iter_in_cycle < change_cycle // 2:
        w_kld = w_kld * (1 / 2 + iter_in_cycle / (change_cycle // 2) / 2)

    # we don't train from discriminator if iter < disc_start
    w_dis = 0.5
    if cur_iter < disc_start:
        w_dis = 0 # cur_iter / disc_start * 0.5
    return w_recon, w_kld, w_dis

def sinusoidal_embedding(pos_len, time_emb_size):
    '''
        Positional embedding is introduced in the paper 'Attention is All You Need.' 
        This concept addresses the issue in NLP tasks where neural networks lack 
        knowledge of the neuron's position in the sequence during self-attention. 
        Position plays a crucial role in performing these tasks. Consequently, 
        the authors proposed the idea of adding positional encoding to each neuron, 
        using a sinusoidal embedding function.

        Sinusoidal embedding function:
        EMB[pos, 2i]  = sin(pos / 10000^(2i/d_model))
        EMB[pos, 2i+1]= cos(pos / 10000^(2i/d_model))
    '''

    # Create all zeros vector.
    time_embedding = torch.zeros([pos_len, time_emb_size])

    # div = exp(2i / d_model * -log(10000) ) = 1 / (10000^(2i / d_model))
    div_term = torch.exp(torch.arange(0, time_emb_size, 2) * (-math.log(10000.0) / time_emb_size))
    time_pos = torch.unsqueeze(torch.arange(0, pos_len), -1)

    # use sin for 2i and cos for 2i+1
    time_embedding[:, 0::2] = torch.sin(time_pos * div_term)
    time_embedding[:, 1::2] = torch.cos(time_pos * div_term)

    # This vector shouldn't be backprop so we set requies grad = False
    time_embedding.requires_grad = False
    return time_embedding

def gaussian(window_size, sigma):
    # Do a 1-d gaussian kernel.
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    # Mulitply two 1d window into 2d window
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    # Mean after doing kernel, and do it per each channel.
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    # Make square of mean
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    # E[x^2] - E[x]^2 = VAR[x]
    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    # E[xy] - E[x]E[y] = STD[xy]
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    # C1, C2 is an hyperparameters.
    # This parameters can be find in wikipedia.
    # https://en.wikipedia.org/wiki/Structural_similarity
    C1 = 0.01**2
    C2 = 0.03**2

    # SSIM score
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def ssim(img1, img2, window_size = 11, size_average = True):
    # Reference github: Po-Hsun-Su/pytorch-ssim

    # Use window_size=11 in most case.
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

if __name__ == '__main__':
    # visualize weight schedaler
    import matplotlib.pyplot as plt
    import numpy as np
    T = np.arange(0, 2000, 1)
    M = np.array([weight_scheduler(t) for t in T]).T

    plt.plot(T, M[0], c='red', label="w_recon")
    plt.plot(T, M[1], c='orange', label="w_perceptual")
    plt.plot(T, M[2], c='blue', label="w_kld")
    plt.plot(T, M[3], c='purple', label="w_disc")
    plt.legend()
    plt.show()
