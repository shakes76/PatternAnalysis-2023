import torch
import albumentations as album
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modules import UNET
from utils import (get_loaders,
                    check_accuracy,
                    save_predictions_as_imgs,
                    make_folder_if_not_exists)
import matplotlib.pyplot as plt

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 256  # 1280 originally
IMAGE_WIDTH = 256  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False

TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"
TEST_IMG_DIR = "data/test_images/"
TEST_MASK_DIR = "data/test_masks/"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    """
    Trains the given model for one epoch using the provided data loader and loss function.

    :param loader: DataLoader providing batches of training data.
    :param model: The neural network model to train.
    :param optimizer: Optimizer for updating model weights.
    :param loss_fn: Loss function to compute training loss.
    :param scaler: Gradient scaler for mixed precision training.
    """
    total_steps = len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr = LEARNING_RATE, 
                               steps_per_epoch = total_steps, epochs = NUM_EPOCHS)
    # Initialize tqdm progress bar
    loop = tqdm(loader)
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        losses.append(loss.item())


class diceLoss(torch.nn.Module):
    """
    diceLoss class computes the Dice loss, which is often used for binary segmentation tasks.
    """
    def init(self):
        """
        Constructor for the diceLoss class.
        """
        super(diceLoss, self).init()
    def forward(self,pred, target):
        """
        Calculates and returns the Dice loss between the predictions and target.

        :param pred: Model's output predictions.
        :param target: Ground truth/target tensor.
        
        :return: The computed Dice loss value.
        """
        smooth = 1e-5
        #Smooth value exist to prevent division by 0

        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )   



def main():
    global losses 
    losses = []
    #Creates saved images folder if they dont exist.
    make_folder_if_not_exists("saved_images")

    train_transform = album.Compose([album.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                                    album.Rotate(limit=35, p=1.0),
                                    album.HorizontalFlip(p=0.5),
                                    album.VerticalFlip(p=0.1),
                                    album.Normalize(mean=[0.0, 0.0, 0.0],
                                                    std=[1.0, 1.0, 1.0],
                                                    max_pixel_value=255.0,),
                                    ToTensorV2(),],)

    val_transforms = album.Compose([album.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                                    album.Normalize(
                                        mean=[0.0, 0.0, 0.0],
                                        std=[1.0, 1.0, 1.0],
                                        max_pixel_value=255.0,),
                                    ToTensorV2(),],)


    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = diceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR,
                                        TRAIN_MASK_DIR,
                                        VAL_IMG_DIR,
                                        VAL_MASK_DIR,
                                        BATCH_SIZE,
                                        train_transform,
                                        val_transforms,
                                        NUM_WORKERS,
                                        PIN_MEMORY,)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()


    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)
        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    #save the model under the name of "model.pth"
    plt.figure(figsize=(10,5))
    plt.plot(losses)
    plt.title('Training Loss over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.grid(True)
    plt.savefig('the_loss_plot.png')
    plt.show()
    FILE = "model.pth"
    torch.save(model.state_dict(), FILE)

if __name__ == "__main__":
    main()