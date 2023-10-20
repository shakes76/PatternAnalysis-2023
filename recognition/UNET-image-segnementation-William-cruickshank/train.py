# this will contain the source code for training, validating, testing and saving the model. The model will be imported from "modules.py" and the data loader should be imported from "dataset.py". it will also plot the losses and metrics of training

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from modules import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 20
NUM_WORKERS = 2
IMAGE_HEIGHT = 192 # 384
IMAGE_WIDTH = 255 # 511
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "recognition/UNET-image-segnementation-William-cruickshank/data/train_images"
TRAIN_MASK_DIR = "recognition/UNET-image-segnementation-William-cruickshank/data/train_masks/"
VAL_IMG_DIR = "recognition/UNET-image-segnementation-William-cruickshank/data/val_images/"
VAL_MASK_DIR = "recognition/UNET-image-segnementation-William-cruickshank/data/val_masks/"
TEST_IMG_DIR = "recognition/UNET-image-segnementation-William-cruickshank/data/test_images/"
TEST_MASK_DIR = "recognition/UNET-image-segnementation-William-cruickshank/data/test_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
        #print("Min target value:", targets.min().item())
        #print("Max target value:", targets.max().item())

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #print(predictions)
            predictions = torch.sigmoid(predictions)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    return loss

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    train_loader, val_loader, test_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        TEST_IMG_DIR,
        TEST_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    
    losses = []
    dices = []
    
    for epoch in range(NUM_EPOCHS):
        losses.append(train_fn(train_loader, model, optimizer, loss_fn, scaler))

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        dices.append(check_accuracy(val_loader, model, device=DEVICE))

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_test_images/", device=DEVICE
        )
        print(losses)
        print(dices)
    check_accuracy(test_loader, model, device=DEVICE)
    save_predictions_as_imgs(
            test_loader, model, folder="saved_test_images/", device=DEVICE
        )

if __name__ == "__main__":
    main()