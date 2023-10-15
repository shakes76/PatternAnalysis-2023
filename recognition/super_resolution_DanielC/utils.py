import torchvision.transforms as transforms

# IO-paths
AD_train_dir = "F:/COMP3710/data/AD_NC/train/train_AD"
NC_train_dir = "F:/COMP3710/data/AD_NC/train/train_NC"
AD_test_dir = "F:/COMP3710/data/AD_NC/test/test_AD"
NC_test_dir = "F:/COMP3710/data/AD_NC/test/test_NC"
model_path = "F:/COMP3710/model/model.pt"

# Hyper-parameters
num_epochs = 100
learning_rate = 0.001
batch_size = 30

out_channels = 256

# Downsample-parameters 
downsample_size = (60,64)

def resize_tensor(tensor):
    transform = transforms.Compose([transforms.Resize(downsample_size)])

    return transform(tensor)
