from utils import *
from modules import Network
from dataset import load_data
import torchvision.utils as vutils
import torch
import torchvision
import torchvision.utils as vutils
from dataset import *

if __name__ == '__main__':
    
    test_loader = load_data(test_path)

    # Load the pretrained model
    model = Network(upscale_factor=upscale_factor, channels=channels)

    model.load_state_dict(torch.load(trained_path, map_location=device))

    model.to(device)

    model.eval()

    with torch.no_grad():
        for input, label in test_loader:

            down_sampled_img = down_sample(input).to(device)
            input = input.to(device)

            output = model(down_sampled_img)

            break
    
    down_scaled_img_upscaled = up_sample(down_sampled_img)

    images = torch.cat((input, down_scaled_img_upscaled, output))
    image = vutils.make_grid(images, padding=2, normalize=True)
    transform = torchvision.transforms.ToPILImage()
    img = transform(image)
    img.show()