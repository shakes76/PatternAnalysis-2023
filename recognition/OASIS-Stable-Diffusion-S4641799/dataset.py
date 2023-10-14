import utils
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define data transformations
transform = transforms.Compose([
    transforms.Resize((utils.IMAGE_SIZE, utils.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),                      
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_dir = "keras_png_slices_data/"
subfolder = "keras_png_slices_"
folder_suffix = "seg_"
folder_type = {"test", "train", "validate"}

def create_data_loader(dataset_type):
    assert dataset_type in folder_type
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=utils.BATCH_SIZE, shuffle=True, )
    return loader