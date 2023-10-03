import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

root_path = 'data/keras_png_slices_data'

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),                      
    transforms.ToTensor(), # Data is scaled into [0, 1]
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
])

# Define batch size of data
batch_size = 32

# Custom OASIS brain dataset class referenced from ChatGPT3.5: how to create custom dataset class for OASIS
class OASISDataset(Dataset):
    def __init__(self, root, label_path, transform=None):
        self.root_dir = root
        self.label_path = label_path
        self.transform = transform

        # List all image files and label files in root and label path directory
        self.image_files = [f for f in os.listdir(self.root_dir) if os.path.isfile(os.path.join(self.root_dir, f))]
        self.label_files = [f for f in os.listdir(self.label_path) if os.path.isfile(os.path.join(self.label_path, f))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.root_dir, self.image_files[index])
        label_path = os.path.join(self.label_path, self.label_files[index])
        # Load and preprocess the image
        image, label = self.load_image(image_path, label_path)
        return image, label
    
    def load_image(self, path1, path2):
        image = Image.open(path1)
        label = Image.open(path2)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label
    
    
# Specifying paths to train, test and validate directories
train_data = OASISDataset(root=f'{root_path}/keras_png_slices_train', label_path=f'{root_path}/keras_png_slices_seg_train', transform=transform)
test_data = OASISDataset(root=f'{root_path}/keras_png_slices_test', label_path=f'{root_path}/keras_png_slices_seg_test', transform=transform)
validate_data = OASISDataset(root=f'{root_path}/keras_png_slices_validate', label_path=f'{root_path}/keras_png_slices_seg_validate', transform=transform)

# Create data loaders for each set
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True) # Image shape [32, 1, 224, 224]
test_loader = DataLoader(test_data, batch_size=batch_size)
validate_loader = DataLoader(validate_data, batch_size=batch_size)