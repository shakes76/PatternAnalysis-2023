from modules import *
import matplotlib.pyplot as plt


def load_image():
    # Loading Generator network
    netG = StyleGANGenerator(z_dim, init_channels, init_resolution, 'cpu')
    netG.load_state_dict(torch.load("model_gen.pt"))
    netG.eval()

    # Generating output
    noise = torch.randn(batch_size, 1, 1, z_dim)
    output = netG(noise)
    output = output[0].detach().apply_(lambda x: (x / max(output.max(), -output.min()) + 1) / 2.0).permute(1, 2, 0)

    # Displaying image
    plt.imshow(output.cpu().detach().numpy())
    plt.show()
