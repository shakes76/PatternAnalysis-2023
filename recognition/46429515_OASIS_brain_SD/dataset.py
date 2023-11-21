import os
import utils
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image


# Define data transformations
transform = transforms.Compose([
    transforms.Resize((utils.IMAGE_SIZE, utils.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),                      
    transforms.ToTensor(), # Data is scaled into [0, 1]
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
])


# Custom OASIS brain dataset class referenced from ChatGPT3.5: how to create custom dataset class for OASIS
class OASISDataset(Dataset):
    def __init__(self, root, label_path, transform=None):
        self.root_dir = root
        self.label_path = label_path
        self.transform = transform
        self.image_paths = self._make_dataset() 
     
        
    def _make_dataset(self):
        image_paths = []
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".png"):
                    image_paths.append(os.path.join(subdir, file))
        return image_paths
    
        
    def __len__(self):
        return len(self.image_paths)
    
    
    def __getitem__(self, index):
        
        image_path = self.image_paths[index]
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        # Load corresponding label from label directory
        label_filename = os.path.basename(image_path).replace("case_", "seg_")
        label_path = os.path.join(self.label_path, label_filename)
        
        label_image = Image.open(label_path)
        
        if self.transform:
            label_image = self.transform(label_image)
        
        return image, label_image
    
    
# Specifying paths to train, test and validate directories
train_data = OASISDataset(root=f'{utils.root_path}/keras_png_slices_train', 
                          label_path=f'{utils.root_path}/keras_png_slices_seg_train', transform=transform)
test_data = OASISDataset(root=f'{utils.root_path}/keras_png_slices_test', 
                         label_path=f'{utils.root_path}/keras_png_slices_seg_test', transform=transform)
combined_data = ConcatDataset([train_data, test_data])
validate_data = OASISDataset(root=f'{utils.root_path}/keras_png_slices_validate', 
                             label_path=f'{utils.root_path}/keras_png_slices_seg_validate', transform=transform)

# Create data loaders for the combined data and validate data
data_loader = DataLoader(combined_data, batch_size=utils.BATCH_SIZE, shuffle=True, drop_last=True) # Image shape [32, 1, 128, 128]
validate_loader = DataLoader(validate_data, batch_size=utils.BATCH_SIZE)