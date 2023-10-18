from modules import ImprovedUNET, DiceLoss
from dataset import ISICdataset
from utilities import get_data_from_url, train, DSC, accuracy
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import wandb

wandb.init(project="Lesion_detection", name="UNET")

print('got here')
#get_data_from_url('ISIC_data', '1vxd1KBIYa3zCAVONkacdclsWUAxhWLho')
TRAIN_DIR = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2'
TRUTH_DIR = '/home/groups/comp3710//ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2'


train_transform = A.Compose(
        [
            A.Resize(height=256, width=256),
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


full_dataset = ISICdataset(
        image_dir=TRAIN_DIR,
        truth_dir=TRUTH_DIR,
        transform=train_transform,
    )

#hyperparameters
N_CHANNELS = 3
N_CLASSES = 1
BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(
    full_dataset,
    batch_size=BATCH_SIZE,
    num_workers=1,
    pin_memory=True,
    shuffle=True,
)

model = ImprovedUNET(N_CHANNELS, N_CLASSES)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = DiceLoss()

model = model.to(DEVICE)
print(model)
print(DEVICE)

for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, loss_fn, DEVICE)
    acc, dice = accuracy(model, train_loader, DEVICE)
    wandb.log({'Epoch': epoch, 'Accuracy': acc, 'Dice Similarity': dice})
    

