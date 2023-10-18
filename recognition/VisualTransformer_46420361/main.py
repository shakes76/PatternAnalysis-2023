"""The primary function where hyperparameters and variables can be tuned"""
from train import *
from modules import *


def main():
    # data variables
    model_name = 'revert_to_crop.pth'
    root = '/home/callum/AD_NC/'
    
    # hyperparameters
    epochs = 10
    learning_rate = 0.001
    weight_decay = 0.0001
    
    device  = get_device()

    model = create_model(image_size=image_size,
                         channels=channels,
                         patch_size=patch_size,
                         embedding_dims=embedding_dims,
                         num_heads=num_heads,
                         device=device)
    
    train_model(model=model,
                root=root,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                device=device)
    
    save_model(model, model_name)

        
if __name__ == '__main__':
    main()