from torchvision import transforms, datasets
from torchvision.transforms import Normalize, Compose, ToTensor
from torch.utils.data import DataLoader
from train import BATCH_SIZE


# Make adjustment to the original pictures
transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((128, 128)),  # Do not use Resize if the GPU memory is enough,
    # this can affect the generated image resolution
    ToTensor()
])

# Rename the image folder path manually if necessary, ADNI dataset is represented as an example for this model
train_img_dir = "D:/ADNI_AD_NC_2D/AD_NC/train"
train_dataset = datasets.ImageFolder(root=train_img_dir, transform=transform)
training_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

