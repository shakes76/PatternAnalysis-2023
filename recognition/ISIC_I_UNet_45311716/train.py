import torch
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from modules import ImprovedUNet
from dataset import UNetData

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Path to image directory
data_path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ISIC2018/' 
# Location to save trained model
save_path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/ImprovedUNet.pt'
# Location to save training and validation loss plot
plot_path = 'C:/Users/mlamt/OneDrive/UNI/2023/Semester 2/COMP3710/Code/data/Train_and_Valid_Loss_Plot.png'

# Hyper-parameters
num_epochs = 15
learning_rate = 1e-3
image_height = 512 
image_width = 512
batch_size = 16

# Following function is from github:
# Reference: https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

def print_plot(train_loss, valid_loss):
    # plot training and validaition Loss's
    plt.plot(train_loss, label="Training Loss")
    plt.plot(valid_loss, label="Validation Loss")

    # plot names
    plt.title("Training Losses")
    plt.xlabel("Epoch Number")
    plt.ylabel("Average Loss")

    plt.ylim(0,1)
    plt.legend()

    plt.savefig(plot_path)

def validation(model, valid_data):
    total_correct = 0 # total number of correct pixels
    total_pixels = 0 # total number of pixels
    valid_loss = 0 # total validation loss

    model.eval() # model to evaluation mode
    
    # disable gradient calculations
    with torch.no_grad():
        for image, mask in valid_data:
            image = image.to(device)
            mask = mask.float().to(device) 

            # model prediction in binary form
            pred = (model(image) > 0.5).float()
            
            total_correct += (pred == mask).sum() 
            total_pixels += torch.numel(pred)

            valid_loss += dice_loss(pred, mask).detach().cpu().numpy()

    accuracy = total_correct/total_pixels*100.0
    accuracy = "{:.2f}".format(accuracy)
    v_loss = valid_loss/len(valid_data)
    dice_score = 1 - v_loss
    
    model.train() # model to train mode

    return accuracy, dice_score, v_loss

def main():
    # Improved UNet model
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)   

    # Training data
    data = UNetData(path=data_path, height=image_height, width=image_width, batch=batch_size)
    train_data = data.get_train_loader()
    valid_data = data.get_valid_loader()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    total_step = len(train_data)

    print(" - - Start Training - - ")
    start = time.time()
    train_loss = []
    valid_loss = []
    # Gradient scaler
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epochs):
        total_t_loss = 0
        for batch in train_data:
            image, mask = batch

            image = image.to(device=device)
            mask = mask.float().to(device=device) 

            # automatic mixed-precision training
            with torch.cuda.amp.autocast():
                predictions = model(image)
                loss = dice_loss(predictions, mask) 
                total_t_loss += loss.detach().cpu().numpy()

            # Backpropagation 
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        t_loss = total_t_loss/total_step
        train_loss.append(t_loss)
        # validation
        accuracy, dice_score, v_loss = validation(model, valid_data)
        valid_loss.append(v_loss)
        print (f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {t_loss},\n Dice Score: {dice_score}, Accuracy: {accuracy}, Validation Loss: {v_loss}")

    # save model
    torch.save(model, save_path)

    end = time.time()
    elapsed = end - start
    print(f"Total time: {elapsed/60} min")

    print_plot(train_loss, valid_loss)

if __name__ == "__main__":
    main()