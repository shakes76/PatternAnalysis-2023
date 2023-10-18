import utils, os
from torchvision import transforms
from torch.utils.data import DataLoader, dataset
from PIL import Image

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((utils.IMAGE_SIZE, utils.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_dir = "keras_png_slices_data/"
subfolder = "keras_png_slices_"
folder_suffix = "seg_"
folder_type = {"test", "train", "validate"}

class FolderLoader(dataset.Dataset):
    def __init__(self, root, transform=None, includeSeg = False):
        self.root_dir = root
        self.transform = transform
        self.includeSeg = includeSeg
        self.image_paths = self._make_dataset()


    def _make_dataset(self):
        image_paths = []
        directories = [self.root_dir]
        if self.includeSeg:
            directories.append(self.root_dir.replace('data/' + subfolder, 'data/' + subfolder+folder_suffix))
        for directory in directories:
            for subdir, _, files in os.walk(directory):
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
        return image

def create_data_loader(dataset_type):
    assert dataset_type in folder_type
    dataset = FolderLoader(root=data_dir+subfolder+dataset_type, transform=transform, includeSeg=True)
    loader = DataLoader(dataset, batch_size=utils.BATCH_SIZE, shuffle=True)
    return loader