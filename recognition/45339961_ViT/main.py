import torch
from torchvision import transforms
from dataset import create_datasets
from utils import show_patched_image

def main():
    IMAGE_WIDTH = 224
    PATCH_SIZE = 16

    train_transform = transforms.Compose([
        # transforms.CenterCrop(IMAGE_CROP),
        transforms.Resize((IMAGE_WIDTH, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    root_dir = "C:/Users/Jacqu/Downloads/AD_NC"
    train_data, _, _ = create_datasets(root_dir=root_dir,
                                                        train_transform=train_transform,
                                                        test_transform=test_transform,
                                                        datasplit=0.8)

    img_tensor, label = train_data[0]
    img_numpy = img_tensor.squeeze().numpy()

    show_patched_image(img_numpy, IMAGE_WIDTH, PATCH_SIZE)

if __name__ == '__main__':
    main()

# if __name__ == "__main__":
#     BATCH_SIZE = 32
#     IMAGE_WIDTH = 192

#     train_transform = transforms.Compose([
#         transforms.Resize((IMAGE_WIDTH, IMAGE_WIDTH)),
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.Resize((IMAGE_WIDTH, IMAGE_WIDTH)),
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor(),
#     ])

#     root_dir = "C:/Users/Jacqu/Downloads/AD_NC"

#     train_data, valid_data, test_data = create_datasets(root_dir=root_dir, 
#                                                         train_transform=train_transform,
#                                                         test_transform=test_transform,
#                                                         datasplit=0.8)
#     print(f"Number of classes: {len(train_data.classes)}")
#     print(f"Class names: {train_data.classes}")

#     data = train_data

#     import random
#     # Randomly select an image and display its details
#     # random_index = random.randint(0, len(train_data) - 1)
#     random_index = 0
#     img_tensor, label = data[random_index]
#     class_name = data.classes[label]

#     # Convert tensor back to PIL Image for visualization
#     to_pil = transforms.ToPILImage()
#     img_pil = to_pil(img_tensor)
#     img_size = img_tensor.shape

#     print(f"Random Image Details:")
#     print(f"Label: {class_name}")
#     print(f"Size: {img_size}")
#     img_path = data.samples[random_index][0]  # Get the path
#     filename = os.path.basename(img_path)
#     print(f"Filename: {filename}")
#     img_pil.show()

#     from collections import Counter

#     # Count occurrences of each label in train_data
#     label_counts = Counter([label for _, label in data.samples])

#     # Print number of images for each class
#     for class_idx, count in label_counts.items():
#         class_name = data.classes[class_idx]
#         print(f"Class: {class_name}, Number of Images: {count}")