import os
import shutil
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
    w_perceptual = 0.5

    # https://github.com/haofuml/cyclical_annealing
    # Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing (NAACL 2019):
    w_kld = 10
    # first half is linear, end half is constant
    iter_in_cycle = cur_iter % change_cycle
    if iter_in_cycle < change_cycle // 2:
        w_kld = w_kld * (1 / 2 + iter_in_cycle / (change_cycle // 2) / 2)

    # we don't train from discriminator if iter < disc_start
    w_dis = 0.5
    if cur_iter < disc_start:
        w_dis = 0 # cur_iter / disc_start * 0.5
    return w_recon, w_perceptual, w_kld, w_dis


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
