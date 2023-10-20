from parameters import *
from modules import *
from torchvision import datasets, transforms as T

def load_data_celeba():
    # Defining Transforms
    transform_train = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(norm_mean, norm_sd),
        T.RandomHorizontalFlip(),
        # T.RandomCrop(32, padding=4, padding_mode='reflect'),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize(norm_mean, norm_sd),
    ])

    transform = T.ToPILImage()

    # Obtaining datasets
    train_dataset = datasets.CelebA(root, "train", transform=transform_train, download=True)  # Set batch to = 10, or 82

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, transform, train_dataset
