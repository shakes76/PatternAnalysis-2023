import os, torch, math
from torchvision.utils import save_image


"""
Utils of functions and global variables needed for other scripts.
"""

# Path to data images
root_path = 'data/keras_png_slices_data'

IMAGE_SIZE = 256 # Power of 2
BATCH_SIZE = 32

# Number of Epochs for training
epochs = 100

DEVICE                  = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE           = 1e-3
LOG_RESOLUTION          = int(math.log2(IMAGE_SIZE))
Z_DIM                   = IMAGE_SIZE
W_DIM                   = IMAGE_SIZE
LAMBDA_GP               = 10

def gradient_penalty(critic, real, fake,device="cpu"):
    BATCH_SIZE, C, H, W = real.shape
    beta = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * beta + fake.detach() * (1 - beta)
    interpolated_images.requires_grad_(True)

    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
 
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty

def get_w(mapping_network, batch_size):
    z = torch.randn(batch_size, W_DIM).to(DEVICE)
    w = mapping_network(z)
    return w[None, :, :].expand(LOG_RESOLUTION, -1, -1)

def get_noise(batch_size):
        noise = []
        resolution = 4

        for i in range(LOG_RESOLUTION):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)
            n2 = torch.randn(batch_size, 1, resolution, resolution, device=DEVICE)

            noise.append((n1, n2))

            resolution *= 2

        return noise
        
def generate_examples(mapping_network, gen, epoch, start_time, n=100):
    for i in range(n):
        img = generate_images(mapping_network, gen)
        if not os.path.exists(f'saved_examples/{start_time}_epoch{epoch}'):
            os.makedirs(f'saved_examples/{start_time}_epoch{epoch}')
        save_image(img, f"saved_examples/{start_time}_epoch{epoch}/img_{i}.png")

def generate_images(mapping_network, gen, n=1):
    gen.eval()
    with torch.no_grad():
        w     = get_w(mapping_network, n)
        noise = get_noise(n)
        img   = gen(w, noise)
    gen.train()
    return img*0.5+0.5

def generate_noise_images(n=1):
    images = []
    for _ in range(n):
        img = get_noise(1)[-1][0][0]
        images.append(img)
    images = torch.stack(images, dim=0)
    return images
