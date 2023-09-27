import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path

# Create paths to images
TRAIN_DATA_PATH = Path("./data/AD_NC/train/")
TEST_DATA_PATH = Path("./data/AD_NC/test/")

#batch size
BATCH_SIZE = 32

# resize images
IMAGE_SIZE = (224, 224) #TODO: change this value to make training faster

def load_data():
    """
    returns the dataloaders for the training and testing along with the class
    labels and class index into the labels
    """
    #create transforms
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE), 
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=0.5, std=0.5, inplace=True),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Normalize(mean=0.5, std=0.5, inplace=True), #TODO: does normalising do anything to the data?
    ])

    # Load images in using ImageFolder
    train_dataset = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=TEST_DATA_PATH, transform=test_transforms)

    train_loader = DataLoader(dataset=train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    
    test_loader = DataLoader(dataset=test_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=False)
    
    class_labels = train_dataset.classes
    class_idx = test_dataset.class_to_idx
    
    return train_loader, test_loader, class_labels, class_idx