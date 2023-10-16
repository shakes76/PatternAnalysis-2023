from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.cuda
import os
from PIL import Image

# Torch configuration
seed = 42
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resize_size = (128, 128)

# Rangpur = 0, Home PC = 1, Laptop = 2
device_config = 1
if device_config == 0:
    train_path = '/home/groups/comp3710/OASIS/keras_png_slices_train/'
    test_path = '/home/groups/comp3710/OASIS/keras_png_slices_test/'
    validation_path = '/home/groups/comp3710/OASIS/keras_png_slices_validate/'
elif device_config == 1:
    train_path = 'C:/Users/PC User/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data/keras_png_slices_train/'
    test_path = 'C:/Users/PC User/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data/keras_png_slices_test/'
    validation_path = 'C:/Users/PC User/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data/keras_png_slices_validate/'
elif device_config == 2:
    train_path = 'C:/Users/SAM/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data/keras_png_slices_train/'
    test_path = 'C:/Users/SAM/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data/keras_png_slices_test/'
    validation_path = 'C:/Users/SAM/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data/keras_png_slices_validate/'

roots = 'C:/Users/PC User/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data'

class OASISDataLoader:
    def __init__(self, batch_size, image_size=resize_size):
        self.batch_size = batch_size
        self.device_config = 1
        if self.device_config == 0: # rangpur
            root_dir = '/home/groups/comp3710/OASIS/'
        elif self.device_config == 1:   # home pc
            root_dir = 'C:/Users/PC User/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data/'
        elif self.device_config == 2:   # laptop
            root_dir = 'C:/Users/SAM/Desktop/COMP3710/PatternAnalysis-2023/dataset/keras_png_slices_data/'

        train_dir = os.path.join(root_dir, "keras_png_slices_train")
        test_dir = os.path.join(root_dir, "keras_png_slices_test")
        validation_dir = os.path.join(root_dir, "keras_png_slices_validate")

        transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Resize(image_size),
            transforms.Grayscale()
        ])

        self.train = OASISDataset(root_dir=train_dir, transform=transformer)
        self.test = OASISDataset(root_dir=test_dir, transform=transformer)
        self.val = OASISDataset(root_dir=validation_dir, transform=transformer)

    def get_dataloaders(self):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(self.val, batch_size=self.batch_size, shuffle=True)

        return train_loader, test_loader, validation_loader


class OASISDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.file_list = os.listdir(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(file_path)

        if self.transform:
            image = self.transform(image)

        return image