import PIL
import numpy as np
import torch
from torchvision.transforms import ToPILImage, ToTensor
from dataset import load_data
from modules import Network

model = Network(upscale_factor=4, channels=1)
model.load_state_dict(torch.load('./Trained_Model.pth'))
model.eval()

def upscale_image(img_path):
    img = PIL.Image.open(img_path).convert('YCbCr')
    y, cb, cr = img.split()
    input = ToTensor()(y).view(1, -1, y.size[1], y.size[0])
    out = model(input)
    out_img_y = ToPILImage()(out[0].detach().cpu())
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
    return out_img

result = upscale_image('./sample_path.jpg')
result.save('./output.jpg')
