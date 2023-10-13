import torchvision.transforms as transforms
import torch.nn as nn
# IO-paths
AD_train_dir = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710_Report/data/AD_NC/train/AD_train"
NC_train_dir = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710_Report/data/AD_NC/train/NC_train"
AD_test_dir = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710_Report/data/AD_NC/test/AD_test"
NC_test_dir = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710_Report/data/AD_NC/test/NC_test"
model_path = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710_Report/model/model.pt"

# Hyper-parameters
num_epochs = 1
learning_rate = 0.001
batch_size = 200

# Downsample-parameters 
downsample_size = (60,64)

def resize_tensor(tensor):
    transform = transforms.Compose([transforms.Resize(downsample_size)])

    return transform(tensor)
