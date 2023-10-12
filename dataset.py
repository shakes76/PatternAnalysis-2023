import os
import torch
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split

# Define the data directory
data_dir = "AD_NC/"

def test():
    # Create the ImageFolder dataset
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, "test"))
    # Iterate through the dataset and print labels
    image, label = dataset[4]
    print(f"Image: {image}, Label: {label}")

def get_loaders():
    # Define data transformations (resize, normalize, etc.) with data augmentation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize images to a consistent size
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Randomly rotate the image up to 10 degrees
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values (adjust mean and std as needed)
    ])

    # Create the ImageFolder dataset for training
    train_data = datasets.ImageFolder(root=os.path.join(data_dir, "train"), transform=transform)

    # Create the ImageFolder dataset for testing
    test_data = datasets.ImageFolder(root=os.path.join(data_dir, "test"), transform=transform)

    # Define the train-test split ratio (e.g., 80% train, 20% test)
    train_size = 0.8

    # Split the training dataset into training and validation sets
    train_data, validation_data = train_test_split(train_data, train_size=train_size, test_size=1 - train_size, shuffle=True, random_state=42)

    # Create data loaders with reduced batch size and multi-processing
    batch_size = 64  # Adjust as needed
    num_workers = 4  # Use multiple workers for data loading

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    
    return train_loader, validation_loader, test_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader = get_loaders()
    print("done da sdasd")
