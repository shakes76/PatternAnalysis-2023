import torch
from torchvision import transforms
from dataset import create_datasets, create_dataloaders
from utils import show_patched_image
from modules import ViT
from torchinfo import summary
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from train import train
from predict import predict, test

def main():
    PATCH_SIZE = 16 # P
    IMAGE_CROP = 240
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = IMAGE_WIDTH
    IMAGE_CHANNELS = 1
    BATCH_SIZE = 64
    NUM_HEADS = 6
    NUM_LAYERS = 2
    MLP_DROPOUT = 0.1
    ATTN_DROPOUT = 0.0
    EMBEDDING_DROPOUT = 0.1
    MLP_SIZE = 768
    NUM_CLASSES = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    N_EPOCHS = 5
    LR = 0.001

    # EMBEDDING_DIMS = IMAGE_CHANNELS * PATCH_SIZE**2 # Hidden Size D
    EMBEDDING_DIMS = 24

    #the image width and image height should be divisible by patch size. This is a check to see that.
    assert IMAGE_WIDTH % PATCH_SIZE == 0 and IMAGE_HEIGHT % PATCH_SIZE ==0 , print("Image Width is not divisible by patch size")

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_WIDTH, IMAGE_WIDTH)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    root_dir = "C:/Users/Jacqu/Downloads/AD_NC"
    # train_data, , _ = create_datasets(root_dir=root_dir,
    #                                                     train_transform=train_transform,
    #                                                     test_transform=test_transform,
    #                                                     datasplit=0.8)
    # img_tensor, label = train_data[0]
    # img_numpy = img_tensor.squeeze().numpy()
    # show_patched_image(img_numpy, IMAGE_WIDTH, PATCH_SIZE)

    train_loader, valid_loader, test_loader = create_dataloaders(root_dir, train_transform, test_transform, batch_size=BATCH_SIZE, datasplit=0.8)

    model = ViT(
            img_size=IMAGE_WIDTH,
            in_channels = IMAGE_CHANNELS,
            patch_size = PATCH_SIZE,
            embedding_dim = EMBEDDING_DIMS,
            num_transformer_layers = NUM_LAYERS,
            mlp_dropout = MLP_DROPOUT,
            attn_dropout = ATTN_DROPOUT,
            embedding_dropout=EMBEDDING_DROPOUT,
            mlp_size = MLP_SIZE,
            num_heads = NUM_HEADS,
            num_classes = NUM_CLASSES).to(device)

    summary(model=model,
        input_size=(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH), # (batch_size, num_patches, embedding_dimension)
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

    optimizer = Adam(model.parameters(), 
                lr=LR)
                # weight_decay=0.1,
                # betas=(0.9, 0.999)) # Based on the paper
    criterion = CrossEntropyLoss()

    train_accuracies, valid_accuracies, train_losses, valid_losses = train(model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_epochs=N_EPOCHS)
    
    test(model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device)

    predict(model=model, 
            dataloader=test_loader, 
            device=device)


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