#dataset module
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ISICdataset(Dataset):
    def __init__(self, image_dir, truth_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.truth_dir = truth_dir
        self.transform = transform
        self.target_size = target_size
        self.images = os.listdir(image_dir)

        self.images = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        truth_path = os.path.join(self.truth_dir, self.images[index].replace('.jpg', '_segmentation.png'))
        
        # Open and resize the image to the target size
        image = Image.open(img_path).convert('RGB')
        image = transforms.Resize(self.target_size)(image)

        # Open and resize the truth mask to the target size
        truth = Image.open(truth_path).convert('L')
        truth = transforms.Resize(self.target_size)(truth)

        if self.transform is not None:
            augmentations = self.transform(image=np.array(image), mask=np.array(truth))
            image = augmentations['image']
            truth = augmentations['mask']

        return image, truth

    def __len__(self):
        return len(self.images)