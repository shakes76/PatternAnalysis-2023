"""For making predictions on the test dataset"""
from train import *
from dataset import load_dataloaders

root = '/home/callum/AD_NC/'
image_size = 256
crop_size = 192
batch_size = 64

dataloader = load_dataloaders(root=root,
                              image_size=image_size,
                              crop_size=crop_size,
                              batch_size=batch_size)

model_name = ''
model = load_model(model_name=model_name)
predict(model=model,
        dataloader=dataloader,
        num_samples=5)
