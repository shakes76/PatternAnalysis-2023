"""The primary function where hyperparameters and variables can be tuned"""
from train import *
from modules import *
from utils import *


def main():
    """main function for tuning and training model"""
    # data variables
    model_name = 'my_model.pth'
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
    
    training_accuracies, training_losses, validation_accuracies, validation_losses = train_model(model=model,
                root=root,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                epochs=epochs,
                device=device)
    
    plot_accuracies_and_losses(training_accuracies, training_losses, validation_accuracies, validation_losses, epochs)
    
    save_model(model, model_name)

        
if __name__ == '__main__':
    main()