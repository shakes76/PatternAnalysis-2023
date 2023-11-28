import torch, utils, modules, pickle, dataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torchvision import transforms

"""
Author name: Eli Cox
File name: predict.py
Last modified: 22/11/2023
Generate and save images from the supplied trained model
"""

gen                 = modules.Generator(utils.LOG_RESOLUTION, utils.W_DIM).to(utils.DEVICE)
mapping_network     = modules.MappingNetwork(utils.Z_DIM, utils.W_DIM).to(utils.DEVICE)
load_best = False
compare = True

# Load model
start_time = 1697679243.0313046
if load_best:
    gen.load_state_dict(torch.load(f"best_gen_{start_time}.pth"))
    mapping_network.load_state_dict(torch.load(f'best_map_{start_time}.pth'))
    with open(f'best_checkpoint_{start_time}.pickle', 'rb') as handle:
        epoch, _, _ = pickle.load(handle)
else:
    gen.load_state_dict(torch.load(f"latest_gen_{start_time}.pth"))
    mapping_network.load_state_dict(torch.load(f'latest_map_{start_time}.pth'))
    with open(f'latest_checkpoint_{start_time}.pickle', 'rb') as handle:
        epoch, _, _ = pickle.load(handle)

# Load validation data to compare
if compare:
    loader = dataset.create_data_loader("validate", transforms.Compose([
        transforms.Resize((utils.IMAGE_SIZE, utils.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(3),
        transforms.ToTensor(),
    ]))
    valid = next(iter(loader))
    # Generate comparison noise
    noise = utils.generate_noise_images(32)
predicted = utils.generate_images(mapping_network, gen, 32)

# Generate MRIs in one image
predicted = make_grid(predicted, nrow=8)
predicted = predicted.cpu()
if compare:
    valid = make_grid(valid, nrow=8)
    noise = make_grid(noise, nrow=8)
    noise = noise.cpu()
    img = make_grid([predicted, valid], nrow = 2)
    save_image(valid, f"{start_time}_comparison.png")
    save_image(img, f"{start_time}_combined.png")
    save_image(noise, f"{start_time}_noise.png")
save_image(predicted, f"{start_time}_predicted.png")


# Show images
if compare:
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    plt.imshow(noise.permute(1, 2, 0))
    plt.show()
else:
    plt.imshow(predicted.permute(1, 2, 0))
    plt.show()
