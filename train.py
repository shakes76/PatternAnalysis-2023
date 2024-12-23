import dataset as dst
import modules as mm
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

IMAGE_SIZE = 128
NUM_EPOCHS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 0.001
KERAS_PATH = '/home/groups/comp3710/HipMRI_Study_open/keras_slices_data/keras_slices'

NUM_CHANNELS = 1
NUM_CLASSES = 1
NUM_LEVELS = 3

WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9

def dice_similarity_coeff(pred, target, smooth=1e-6):
    '''
    Calculate Dice Similarity Coefficient by calculating
    the number of matched pixels / total pixels.
    Test loss function.
    '''
    pred = torch.sigmoid(pred)  # Convert output into probability
    intersection = (pred * target).sum(dim=(2, 3))  
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))  
    
    dice = (2. * intersection + smooth) / (union + smooth) 
    return dice.mean()

#Visualization of loss
def plot_line_graph(data, title:str, xlabel:str, ylabel:str, filename:str):
    """
    Plot 2D graph with xlabel = n_epoch (train and validation) or n_batch (test)
    and ylabel = corresponding loss value.
    Use savefig to save the image in the directory.
    
    Parameters:
    data (list or array): values to plot
    title (str)
    xlabel (str)
    ylabel (str)
    filename (str): Png file's name to save as.
    """
    # domain 0 to n-1
    x = list(range(len(data)))
    # plotting
    plt.plot(x, data, marker='o')
    # Title and labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.savefig(filename)

def criterion(criteria, outputs, labels, n_classes=6):
    '''
    Calculate the Dice Similarity Coefficient per class.
    Use dst.to_channels() to separate masking for each class and calculate
    one by one.
    '''


def main():

    train_loader, train_set = dst.keras_dataloader(IMAGE_SIZE, '_train/')
    validation_loader, validation_set = dst.keras_dataloader(IMAGE_SIZE, '_validate/')
    test_loader, test_set = dst.keras_dataloader(IMAGE_SIZE, '_test/')
        
    # For Visualization
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []

    #Define UNet model, loss function and update optimizer.
    model = mm.UNet().to(DEVICE)
    criteria = nn.CrossEntropyLoss().to(DEVICE) #BCE Loss accepts softmaxed tensor while CrossEntropy not.
    optimizer = optim.SGD(model.parameters(), weight_decay=WEIGHT_DECAY, lr = LEARNING_RATE, momentum=MOMENTUM)

    for epoch in range(NUM_EPOCHS):
        print(str(epoch + 1) + '/' + str(NUM_EPOCHS) + 'th iteration')
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        for raw, seg in train_loader:
            #Batch index, raw data and segmented data
            imgs, labels = raw.to(DEVICE), seg.to(DEVICE) #(12, 1, 256, 128) (12, 6, 256, 128)
            #(Batch size, Channel number, Width, Height)
            optimizer.zero_grad()
            outputs = model(imgs) #(12, 6, 256, 128)  #Segmented to 6 classes
            loss = criteria(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print('Passed all')
        torch.save('model_state_dict': model.state_dict(), \
                   'optimizer_state_dict': optimizer.state_dict(), \
            f = '/home/Student/s4722435/miniconda3/envs/new_torch/unet_stuff/model.pth')
        model.eval()
        with torch.no_grad():
            for raw, seg in validation_loader:
                imgs, labels = raw.to(DEVICE), seg.to(DEVICE)
                outputs = model(imgs)
                loss = criteria(outputs, labels)
                loss.backward()
                val_loss += loss.item()
        torch.save(model.state_dict(), \
            f = '/home/Student/s4722435/miniconda3/envs/new_torch/unet_stuff/model.pth')
        #Calculate average misclassification (per pixel) rate for train and validation sets.
        train_loss_avg = train_loss / len(train_set)
        val_loss_avg = val_loss / len(val_loss)
        train_loss_list.append(train_loss_avg)
        valid_loss_list.append(val_loss_avg)

    test_loss = 0.0
    dsc = 0.0
    model.eval()
    print('Test set')
    for raw, seg in test_loader:
        imgs, labels = raw.to(DEVICE), seg.to(DEVICE)
        outputs = model(imgs)
        loss = criteria(outputs, labels)
        loss.backward()
        val_loss += loss.item()
        test_loss += loss
        test_loss_list.append(loss)
        dsc = dice_similarity_coeff(outputs, labels)
    test_loss = test_loss // len(test_set)
    dsc = dsc // len(test_set)

    plot_line_graph(train_loss_list, xlabel = 'epoch', ylabel = 'Train Loss', filename = 'train_loss.png')
    plot_line_graph(valid_loss_list, xlabel = 'epoch', ylabel = 'Validation Loss', filename = 'val_loss.png')
    plot_line_graph(test_loss_list, xlabel = 'batch', ylabel = 'Test Loss', filename = 'test_loss.png')
    print('Dice Similarity Coefficient', dsc)
    torch.save(model.state_dict(), \
            f = '/home/Student/s4722435/miniconda3/envs/new_torch/unet_stuff/model.pth')

def main2():
    train_loader, train_set = dst.keras_dataloader(IMAGE_SIZE, '_train/')
    validation_loader, validation_set = dst.keras_dataloader(IMAGE_SIZE, '_validate/')
    test_loader, test_set = dst.keras_dataloader(IMAGE_SIZE, '_test/')
        
    # For Visualization
    train_loss_list = []
    valid_loss_list = []
    test_loss_list = []

    #Define UNet model, loss function and update optimizer.
    model = mm.UNet().to(DEVICE)
    criterion = nn.BCELoss().to(DEVICE)
    optimizer = optim.SGD(model.parameters(), weight_decay=WEIGHT_DECAY, lr = LEARNING_RATE, momentum=MOMENTUM)

    for epoch in range(NUM_EPOCHS):
        print(f"{epoch}/{NUM_EPOCHS}th iteration")
        model.train()
        train_loss = 0.0
    

if __name__ == "__main__":
    main()