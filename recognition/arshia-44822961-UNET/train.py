import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

from dataset import ISICDataset
from modules import ImprovedUnet

# paths to data. 
TRAIN_DATA_PATH  = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2"
TRAIN_MASK_PATH = "/home/groups/comp3710/ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2"
VALID_DATA_PATH = "/home/Student/s4482296/report1/ISIC2018_Validation_Data"
VALID_MASK_PATH = "/home/Student/s4482296/report1/ISIC2018_Task1_Validation_GroundTruth"

# path to store loss graph and metric output file. 
OUTPUT_DIR_PATH = "/home/Student/s4482296/report1"

# Global constants
BATCH_SIZE = 8
LEARNING_RATE = 0.00005

def train(model, train_loader, valid_loader, num_epochs=100, device="cuda"):
    criterion = dice_coefficient
    optimiser = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.985)

    model.to(device)
    model.train()

    # for storing losses 
    training_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        scheduler.step()
        running_loss = 0.0

        for inputs, masks in train_loader:

            inputs, masks = inputs.to(device), masks.to(device)

            optimiser.zero_grad()

            outputs = model(inputs)
            loss = 1 - criterion(outputs, masks) # we want to maximise dice coefficient, 1 is perfct. 
            # also consider appending coefficients here?

            loss.backward()
            optimiser.step()

            running_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch + 1}, Training Loss: {running_loss / len(train_loader)}")

        # Validate the model
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for val_inputs, val_masks in valid_loader:
                val_inputs, val_masks = val_inputs.to(device), val_masks.to(device)
                val_outputs = model(val_inputs)
                val_loss += criterion(val_outputs, val_masks).item()

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(valid_loader)}")

        # store losses. 
        training_losses.append(running_loss / len(train_loader))
        validation_losses.append(val_loss / len(valid_loader))

    return model, training_losses, validation_losses 

def dice_coefficient(y_true, y_pred, eps=10**-8):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + eps) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + 1.)
    return dice

def plot_losses(train_losses, valid_losses, save_dir):
    plt.figure(figsize=(10,5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Losses over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    # Save plot 
    plt.savefig(os.path.join(save_dir, 'losses_plot.png'))
    plt.close()  # Close plot - best prac to save mem


# loads dataset. 
def load_data(img_path, labels_path, transform, batch_size, shuffle=True):
    train_dataset = ISICDataset(img_path, labels_path, transform) # transform dataset 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle) # load into data loader class. 
    return train_loader


if __name__ == "__main__":
    # connect to gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("Warning CUDA not Found. Using CPU")


    # create improvised unet. 
    model = ImprovedUnet() 

    # set up data transform for data 
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.7084, 0.5821, 0.5360], std=[0.1561, 0.1644, 0.1795])
    ])

    # load datasets 
    train_loader = load_data(TRAIN_DATA_PATH, TRAIN_MASK_PATH, data_transform, batch_size=BATCH_SIZE)
    valid_loader = load_data(VALID_DATA_PATH, VALID_MASK_PATH, data_transform, batch_size=BATCH_SIZE)

    # train improvised unet  
    trained_model, training_losses, validation_losses = train(model, train_loader, valid_loader, device=device, num_epochs=10)
    
    # Save the trained model for predictions  
    torch.save(trained_model, "improved_UNET.pth")

    # plot train and validation loss 
    plot_losses(training_losses, validation_losses, save_dir=OUTPUT_DIR_PATH) 
