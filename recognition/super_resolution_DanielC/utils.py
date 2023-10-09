import torchvision.transforms as transforms
import torch.nn as nn
# IO-paths
AD_train_dir = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710 Report/data/AD_NC/train/AD_train"
NC_train_dir = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710 Report/data/AD_NC/train/NC_train"
AD_test_dir = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710 Report/data/AD_NC/test/AD_test"
NC_test_dir = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710 Report/data/AD_NC/test/NC_test"
model_path = "/Users/danielchoi/Documents/UQ/Y2S2/COMP3710 Report/model"

# Hyper-parameters
num_epochs = 10
learning_rate = 0.001
batch_size = 200

def downsample_tensor(tensor):
    transform = transforms.Compose([transforms.Resize((60, 64))])

    return transform(tensor)
