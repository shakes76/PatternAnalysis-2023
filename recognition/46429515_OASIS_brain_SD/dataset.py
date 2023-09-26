import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

root_path = 'data/keras_png_slices_data'

# Define data transformations
transform = transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor()])

# Specifying paths to train, test and validate directories
train_data = ImageFolder(root=f'{root_path}/keras_png_slices_train', transform=transform)
test_data = ImageFolder(root=f'{root_path}/keras_png_slices_test', transform=transform)
validate_data = ImageFolder(root=f'{root_path}/keras_png_slices_validate', transform=transform)

# Create data loaders for each set
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)
validate_loader = DataLoader(validate_data, batch_size=batch_size)