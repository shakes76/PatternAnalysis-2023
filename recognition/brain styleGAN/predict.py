import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from modules import Generator

from torchvision.utils import save_image


"""
Shows example usage of trained model. Print out any results and / or provide visualisations where applicable.
"""

MODEL_PATH = "C:/Users/aleki/saved_examples/generator6.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHANNELS_IMG            = 1
Z_DIM                   = 256
W_DIM                   = 256
IN_CHANNELS             = 256

STEPS = 6



def Process_Image(img):
    img = img*0.5+0.5

    transposed_img = np.transpose(img, (0,2,3,1))[0] #rearrange the columns for rgb

    trans = torchvision.transforms.Lambda(lambda x: x.repeat(1, 1, 3) if x.size(2)==1 else x) #unpack greyscale
    transposed_img = trans(transposed_img)

    #TODO upscale image
   
    return transposed_img



def Predict(gen):
    gen.eval()
    with torch.no_grad():
        noise = torch.randn(1, Z_DIM).to(DEVICE)
        img = gen(noise, 1.0, STEPS).detach().cpu() #TODO 1.0 = alpha??

        save_image(img*0.5+0.5, f"C:/Users/aleki/saved_examples/test.png")

        plt.imshow(Process_Image(img))
        #plt.imshow(torchvision.transforms.functional.to_pil_image(img[0]))
        plt.show()

gen = Generator(Z_DIM, W_DIM, IN_CHANNELS, img_channels=CHANNELS_IMG).to(DEVICE)
gen.load_state_dict(torch.load(MODEL_PATH))
Predict(gen)

#TODO make animation