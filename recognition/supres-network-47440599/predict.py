import torch
import torchvision
import torchvision.utils as vutils
from modules import SubPixel
from dataset import *
from utils import *


if __name__ == "__main__":

    device = torch.device("cpu")


    test_loader = load_data(test_root,test_batchsize)

    # Instantiate the model
    model = SubPixel() 

    # Load the pre-trained model's state dict
    model.load_state_dict(torch.load(load_path, map_location=device))

    model.eval()

    #obtaining the first batch
    with torch.no_grad():
        for input,label in test_loader:

            down_scaled_image = down_sample(input)
            input = input.to(device)
            down_scaled_image = down_scaled_image.to(device)

            output = model(down_scaled_image).detach().cpu()
            break
    
    #Upscaling the down_sampled image with up_scale function so that it can be displayed along side the input and ouput
    down_scaled_image_up = up_sample(down_scaled_image)

    #displaying the images
    images = torch.cat((input,down_scaled_image_up,output))
    image = vutils.make_grid(images, padding=2, normalize = True)
    transform = torchvision.transforms.ToPILImage()
    img = transform(image)
    img.show()
        