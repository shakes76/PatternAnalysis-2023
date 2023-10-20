from utils import *
from modules import Network
import torchvision.utils as vutils
import torch
import torchvision
import torchvision.utils as vutils
from dataset import *

if __name__ == '__main__':
    
    val_loader = load_test_data(test_path)

    # Load the pretrained model
    model = Network(upscale_factor=upscale_factor, channels=channels)

    # Load the trained model
    model.load_state_dict(torch.load(trained_path, map_location=device))

    # Move the model to the GPU
    model.to(device)

    # Set the model to evaluation mode
    model.eval()

    # Get the first batch of images
    with torch.no_grad():
        for input, label in val_loader:

            down_sampled_img = down_sample(input).to(device)
            input = input.to(device)

            output = model(down_sampled_img)

            break

    # Upscale the downsampled image
    down_scaled_img_upscaled = up_sample(down_sampled_img)

    # Concatenate the images
    images = torch.cat((input, down_scaled_img_upscaled, output))
    image = vutils.make_grid(images, padding=2, normalize=True)
    transform = torchvision.transforms.ToPILImage()
    img = transform(image)
    img.save('./figures/results.jpg')