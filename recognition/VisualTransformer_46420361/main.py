"""The primary function where hyperparameters and variables can be tuned"""
from train import *


def main():
    # data variables
    model_name = 'revert_to_crop.pth'
    root = '/home/callum/AD_NC/'
    image_size = 256
    crop_size = 192
    batch_size = 64
    patch_size = image_size // 8
    channels = 1
    embedding_dims = channels * patch_size**2
    patches = (image_size // patch_size)**2
    num_heads = embedding_dims // 64
    num_classes = 2
    
    # hyperparameters
    epochs = 10
    learning_rate = 0.001
    weight_decay = 0.0001

    model = create_model(image_size=image_size,
                         in_channels=channels,
                         patch_size=patch_size,
                         embedding_dims=embedding_dims,
                         num_heads=num_heads,
                         num_classes=num_classes,
                         patches=patches)
    
    train_model(model=model,
                root=root,
                image_size=image_size,
                batch_size=batch_size,
                crop_size=crop_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs)
    
    save_model(model, model_name)

        
if __name__ == '__main__':
    main()