from imports import *

class ImageDataset(Dataset):
    def __init__(self, directory, image_transforms=None):
        self.directory = directory
        self.image_files = sorted(os.listdir(directory))
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.directory, self.image_files[index])
        image = Image.open(image_path).convert("L") # convert to grayscale
        
        # Apply transformations
        if self.image_transforms:
            image = self.image_transforms(image)
        
        return image

def process_dataset(batch_size=8, 
                    is_validation=False, 
                    pin_memory=False,
                    train_dir="/home/groups/comp3710/OASIS/keras_png_slices_train", 
                    test_dir="/home/groups/comp3710/OASIS/keras_png_slices_test", 
                    val_dir="/home/groups/comp3710/OASIS/keras_png_slices_validate"):
    
    # Define image transformations
    image_transforms = Compose([
        Grayscale(),
        ToTensor(), 
        Lambda(lambda t: (t * 2) - 1),
    ])
    
    # Load validation data if doing validation
    if is_validation:
        val_data = ImageDataset(directory=val_dir, image_transforms=image_transforms)
        return DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    
    else:
        # Load training and test data, concatenate them
        train_data = ImageDataset(directory=train_dir, image_transforms=image_transforms)
        test_data = ImageDataset(directory=test_dir, image_transforms=image_transforms)
        combined_data = ConcatDataset([train_data, test_data])

        return DataLoader(combined_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)

# Script to test the dataset loading and visualization
if __name__ == '__main__':
    # Define directories
    image_dir = os.path.expanduser('/home/groups/comp3710/OASIS/keras_png_slices_train')
    save_dir = os.path.expanduser('~/demo_eiji/sd/images')

    # Check if the specified image directory exists
    if os.path.exists(image_dir):
        print(f"Directory exists: {image_dir}")
    else:
        print(f"Directory does not exist: {image_dir}")
        print(f"Current working directory: {os.getcwd()}")

    data_loader = process_dataset(batch_size=4)

    # Visualize a batch of images
    batch = next(iter(data_loader))
    for i, image in enumerate(batch):
        plt.subplot(1, 4, i+1)
        plt.imshow(image.squeeze(0))
        plt.axis('off')
        
    # Save the image
    save_path = os.path.join(save_dir, f'image_{i}.png')
    plt.savefig(save_path)
