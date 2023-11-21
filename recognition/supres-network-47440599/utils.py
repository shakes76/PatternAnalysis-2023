import torchvision.transforms as transforms
from  torch.nn.modules.upsampling import Upsample


#batch sizes
train_batchsize = 128
test_batchsize = 8

#roots and paths
test_root = "./data/AD_NC/test"
train_root = "./data/AD_NC/train"
load_path = "subpixel_model.pth"
saved_path = "subpixel_model2.pth"

#downscale and upscale functions
down_sample = transforms.Compose([transforms.Resize((60,64))])
up_sample = Upsample(scale_factor=4)

#parameters
num_epochs = 100
learning_rate=0.001
workers = 2
