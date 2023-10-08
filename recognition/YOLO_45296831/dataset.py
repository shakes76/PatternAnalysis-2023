from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import os
import cv2

data_path = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2/"
batch_size = 128

class ISICDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = cv2.imread(img_name)
        
        if self.transform:
            image = self.transform(image)

        return image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(1),
    transforms.Resize((256, 256)),
    transforms.Normalize([0.5], [0.5])])

dataset = ISICDataset(root_dir=data_path, transform=transform)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)      