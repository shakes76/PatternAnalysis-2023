import torch
from modules import UNet
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

image_height = 96
image_width = 128
img_path = "data\ISIC2018_Task1-2_Training_Input_x2\ISIC_0000001.jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet(3, 1).to(device)

model.load_state_dict(torch.load("model2"))

model.eval()

train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_height, image_width), antialias=None)
    ])



img = np.array(Image.open(img_path).convert("RGB"))
img = train_transforms(img)
img = img[None, :, :, :]
img = img.to(device)
img = model(img)
torchvision.utils.save_image(img, "ISIC_0000001_segmentation_from_model.png")