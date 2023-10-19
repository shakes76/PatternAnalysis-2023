import torch
import torch.nn as nn
import torch.optim as optim
from modules import ImprovedUNet
from dataset import UNetData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to images
path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ISIC2018/'
    
# Hyper-parameters
num_epochs = 1
learning_rate = 0.1
image_height = 512 
image_width = 512

def validation(model, valid_data):
    total_correct = 0 # total number of correct pixels
    total_pixels = 0 # total number of pixels
    dice_score = 0 # average dice score
    model.eval() # model to evaluation mode

    # disable gradient calculations
    with torch.no_grad():
        for image, mask in valid_data:
            image = image.to(device)
            mask = mask.to(device) 

            # model prediction in binary form
            pred = torch.sigmoid(model(image))
            pred = (pred > 0.5).float()
            
            total_correct += (pred == mask).sum() 
            total_pixels += torch.numel(pred)

            # calculate dice
            dice_score += (2 * (pred * mask).sum()) / ( (pred + mask).sum() + 1e-9)

    accuracy = total_correct/total_pixels*100.0
    accuracy = "{:.2f}".format(accuracy)
    dice = dice_score/len(valid_data)

    model.train() # model to train mode

    return accuracy, dice

def main():
    # Improved UNet model
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)   

    # Training data
    data = UNetData(path=path, height=image_height, width=image_width)
    train_data = data.train_loader
    valid_data = data.valid_data

    # Binary class. loss function and Adam optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_data)
    # Cosine annealing w/ warm restarts lr scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=total_step, 
                                                                    eta_min=num_epochs, verbose=True)

    print(" - - Start Training - - ")
    # Gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        for batch in train_data:
            image, mask = batch

            image = image.to(device=device)
            mask = mask.float().to(device=device) 

            # automatic mixed-precision training
            with torch.cuda.amp.autocast():
                predictions = model(image)
                loss = loss_fn(predictions, mask) 

            # Backpropagation 
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # validation
        accuracy, dice_score = validation()
        print (f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, 
                \n Dice Score: {dice_score}, Accuracy: {accuracy}")



if __name__ == "__main__":
    main()