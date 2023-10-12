import torch
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

# Define the data directory
data_dir = "ADNI"

def test():
    # Create the ImageFolder dataset
    dataset = datasets.ImageFolder(root=data_dir)

    # Iterate through the dataset and print labels
    image, label = dataset[0]
    print(f"Image: {image}, Label: {label}")
    

def get_loaders(data_dir):
    # Define data transformations (resize, normalize, etc.)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a consistent size
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values (adjust mean and std as needed)
    ])

    # Create the ImageFolder dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Define the train-test split ratio (e.g., 80% train, 20% test)
    train_size = 0.8
    validation_size = 0.1  # Allocate 10% of the data for validation

    # Split the dataset into training, validation, and test sets
    train_data, test_data = train_test_split(dataset, train_size=train_size, test_size=1 - train_size, shuffle=True, random_state=42)
    validation_data, test_data = train_test_split(test_data, test_size=validation_size / (1 - train_size), shuffle=True, random_state=42)

    # Create data loaders
    batch_size = 64  # Adjust as needed

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    
    return train_loader, validation_loader, test_loader

if __name__ == "__main__":
    test()
