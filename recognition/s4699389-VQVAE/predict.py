import matplotlib.pyplot as plt
import torch
import torchvision
from dataset import OASISDataLoader

VQVAE_PATH = "./vqvae_model.txt"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 32

save_fig = False
show_one = True

model = torch.load(VQVAE_PATH)
_, test_loader, _ = OASISDataLoader(batch_size=batch_size).get_dataloaders()


for i, test_data in enumerate(test_loader):
    fig, ax = plt.subplots(1, 3, figsize=(8, 3))
    for x in ax.ravel():
        x.set_axis_off()

    test_real = test_data[0]
    test_real = test_real.to(device).view(-1, 1, 128, 128).detach()

    _, decoded_img = model(test_real)
    decoded_img = decoded_img.view(-1, 1, 128, 128).to(device).detach()
    real_grid = torchvision.utils.make_grid(test_real, normalize=True)
    decoded_grid = torchvision.utils.make_grid(decoded_img, normalize=True)
    decoded_grid = decoded_grid.to("cpu").permute(1, 2, 0)
    real_grid = real_grid.to("cpu").permute(1, 2, 0)

    pre_conv = (model.pre_vq_conv(model.encoder(test_real)))
    _, test_quantized, _, indices = model.vq(pre_conv)
    encoding = indices.view(32, 32)
    encoding = encoding.to('cpu')
    encoding = encoding.detach().numpy()

    ax[0].imshow(real_grid)
    ax[0].title.set_text("Real Image")
    ax[1].imshow(encoding)
    ax[1].title.set_text("Codebook Representation")
    ax[2].imshow(decoded_grid)
    ax[2].title.set_text("Decoded Image")
    if save_fig:
        plt.savefig(f"image_representation_{i}.png")
    if show_one:
        plt.show()
        break
    plt.show()