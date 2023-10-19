from modules import ImprovedUNET, DiceLoss, CustomCompose, CustomResize
from dataset import ISICdataset
from torch.utils.data import DataLoader
import wandb
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import torch
from utilities import accuracy

wandb.init(project="Lesion_detection", name="UNET_2")


# Initialize Dataset and Dataloader
image_dir = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2'
truth_dir = '/home/groups/comp3710//ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2'
transform = CustomCompose([CustomResize((256,256))])
train_dataset = ISICdataset(image_dir, truth_dir, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize Model, Loss and Optimizer
n_channels = 3 # RGB Images
n_classes = 1  # Background and Object
device = 'cuda'
model = ImprovedUNET(n_channels, n_classes)
model = model.to(device)
# criterion = DiceLoss()
# optimizer = Adam(model.parameters(), lr=0.001)
criterion = DiceLoss()
lr_init = 0.0001
weight_decay = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_init * (0.985 ** epoch))

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(train_dataloader)
    for image, truth in progress_bar:
        image = image.to(device)
        truth = truth.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(image)
        
        # Calculate loss
        loss = criterion(pred, truth)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        acc, dice_score = accuracy(model, criterion, pred, truth)
        acc = acc/len(train_dataloader)
        dice_score = dice_score/len(train_dataloader)
        epoch_loss += loss.item()
        
        progress_bar.set_description(f'Epoch {epoch+1}, Loss {loss.item()}')
        wandb.log({'Running loss': loss.item(), 'Acc': acc})
    scheduler.step()
    wandb.log({'Epoch_loss': epoch_loss}) 
    print(f'Epoch {epoch+1}, Avg Loss {epoch_loss/len(train_dataloader)}')


# print('got here')
# #get_data_from_url('ISIC_data', '1vxd1KBIYa3zCAVONkacdclsWUAxhWLho')
# TRAIN_DIR = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2'
# TRUTH_DIR = '/home/groups/comp3710//ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2'


# train_transform = A.Compose(
#         [
#             A.Resize(height=256, width=256),
#             A.Rotate(limit=35, p=1.0),
#             A.HorizontalFlip(p=0.5),
#             A.VerticalFlip(p=0.1),
#             A.Normalize(
#                 mean=[0.0, 0.0, 0.0],
#                 std=[1.0, 1.0, 1.0],
#                 max_pixel_value=255.0,
#             ),
#             ToTensorV2(),
#         ],
#     )


# full_dataset = ISICdataset(
#         image_dir=TRAIN_DIR,
#         truth_dir=TRUTH_DIR,
#         transform=train_transform,
#     )

# #hyperparameters
# N_CHANNELS = 3
# N_CLASSES = 1
# BATCH_SIZE = 16
# LEARNING_RATE = 0.01
# EPOCHS = 6
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_loader = DataLoader(
#     full_dataset,
#     batch_size=BATCH_SIZE,
#     num_workers=1,
#     pin_memory=True,
#     shuffle=True,
# )

# model = ImprovedUNET(N_CHANNELS, N_CLASSES)
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# loss_fn = DiceLoss()

# model = model.to(DEVICE)
# print(model)
# print(DEVICE)

# for epoch in range(EPOCHS):
#     train(model, train_loader, optimizer, loss_fn, DEVICE)
#     acc, dice = accuracy(model, train_loader, DEVICE)
#     wandb.log({'Epoch': epoch, 'Accuracy': acc, 'Dice Similarity': dice})
    

