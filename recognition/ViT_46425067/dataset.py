import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import platform

OS = platform.system()
# Create paths to images
if OS == "Windows":
    TRAIN_DATA_PATH = Path("E:/UNI 2023 SEM 2/COMP3710/Lab3/recognition/ViT_46425067/AD_NC/train")
    TEST_DATA_PATH = Path("E:/UNI 2023 SEM 2/COMP3710/Lab3/recognition/ViT_46425067/AD_NC/test")
else:
    TRAIN_DATA_PATH = Path("./AD_NC/train/")
    TEST_DATA_PATH = Path("./AD_NC/test/")


def load_data(batch_size, image_size):
    """
    returns the dataloaders for the training and testing along with the class
    labels and class index into the labels
    """
    #create transforms
    train_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)), 
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop((224,224), scale=(0.8,1.0), ratio=(0.9, 1.1)),
        # transforms.RandomRotation(degrees=30),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.Normalize(mean=(0.1156), std=(0.2198), inplace=True),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=(0.1156), std=(0.2198), inplace=True),
    ])

    # Load images in using ImageFolder
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False)
    
    class_labels = train_dataset.classes
    class_idx = test_dataset.class_to_idx
    
    return train_loader, test_loader, class_labels, class_idx