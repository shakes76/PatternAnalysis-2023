import utils, os
from torchvision import transforms
from torch.utils.data import DataLoader, dataset
from PIL import Image

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((utils.IMAGE_SIZE, utils.IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    #transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

data_dir = "keras_png_slices_data/"
subfolder = "keras_png_slices_"
folder_suffix = "seg_"
folder_type = {"test", "train", "validate"}

class FolderLoader(dataset.Dataset):
    def __init__(self, root, transform=None):
        self.root_dir = root
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

        """seg_image_path = image_path.replace(subfolder, subfolder+folder_suffix)
        seg_image = Image.open(seg_image_path)

        if self.transform:
            seg_image = self.transform(seg_image)"""
        return image#, seg_image

def create_data_loader(dataset_type):
    assert dataset_type in folder_type
    dataset = FolderLoader(root=data_dir+subfolder+dataset_type, transform=transform)
    loader = DataLoader(dataset, batch_size=utils.BATCH_SIZE, shuffle=True)
    len(loader.dataset)
    return loader