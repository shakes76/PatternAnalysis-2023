"""The primary function where hyperparameters and variables can be tuned"""
from train import *
from modules import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def main():
    # data variables
    model_name = 'revert_to_crop.pth'
    root = '/home/callum/AD_NC/'
    root = '/home/groups/comp3710/ADNI/AD_NC/'
    
    # hyperparameters
    epochs = 10
    learning_rate = 0.001
    weight_decay = 0.0001
    
    # device  = get_device()

    # model = create_model(image_size=image_size,
    #                      channels=channels,
    #                      patch_size=patch_size,
    #                      embedding_dims=embedding_dims,
    #                      num_heads=num_heads,
    #                      device=device)
    train_transform = transforms.Compose([
        # CropBrainScan(),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1155], std=[0.2224]) # Calculated values
    ])

    test_transform = transforms.Compose([
        # CropBrainScan(),
        transforms.CenterCrop((crop_size, crop_size)),
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1167], std=[0.2228]) # Calculated values
    ])
    
    train_dataset = ImageFolder(root + 'train', transform=train_transform)
    test_dataset = ImageFolder(root + 'test', transform=test_transform)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
    model = ViT(
        img_size=image_size,
        in_channels=channels,
        patch_size=patch_size,
        embedding_dims=embedding_dims,
        num_heads=num_heads
        ).to(device)
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    model.train()
    
    for epoch in trange(epochs, desc="Training"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")
        
    model.eval() # evaluation mode
    # Test loop
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
    
    # train_model(model=model,
    #             root=root,
    #             learning_rate=learning_rate,
    #             weight_decay=weight_decay,
    #             epochs=epochs,
    #             device=device)
    
    save_model(model, model_name)

        
if __name__ == '__main__':
    main()