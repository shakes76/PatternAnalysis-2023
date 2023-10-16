import torch
import torchvision
import torchvision.utils as vutils
import torchvision.transforms as transforms
from modules import SubPixel
from dataset import *
from  torch.nn.modules.upsampling import Upsample

if __name__ == "__main__":

    device = torch.device("cpu")

    dataroot = "./data/AD_NC/test"

    test_loader = load_data("./data/AD_NC/test")

    # Instantiate the model
    model = SubPixel() 

    # Load the pre-trained model's state dict
    model.load_state_dict(torch.load("subpixel_model.pth", map_location=device))

    model.eval()
    transform = transforms.Compose([transforms.Resize((60,64))])

    with torch.no_grad():
        for input,label in test_loader:

            down_scaled_image = transform(input)
            input = input.to(device)
            down_scaled_image = down_scaled_image.to(device)

            output = model(down_scaled_image).detach().cpu()
            break

    m = Upsample(scale_factor=4)
    down_scaled_image_up = m(down_scaled_image)

    images = torch.cat((input,down_scaled_image_up,output))
    image = vutils.make_grid(images, padding=2, normalize = True)
    transform = torchvision.transforms.ToPILImage()
    img = transform(image)
    img.show()
        