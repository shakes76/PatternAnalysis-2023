from dataset import TripletImageFolder, TripletImageTestFolder, get_datasets
from modules import TripletLoss, TripletNet, TripletNetClassifier
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

print('TripletNetwork')

train_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomCrop(100, padding=4, padding_mode='reflect'),
        transforms.Grayscale(),
    ])
test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Grayscale(),
    ])

batch_size = 32
num_epochs = [35, 100]
# learning_rate = 0.001

train_folder = 'AD_NC/train'
test_folder = 'AD_NC/test'

train_dataset = TripletImageFolder(train_folder, transform=train_transform)
test_dataset = TripletImageTestFolder(test_folder, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = TripletNet().to(device)

# Define a loss function and an optimizer
criterion = TripletLoss(0.5)

# SGD optimiser 1
learning_rate = 0.1
optimiser_one = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# Piecwise Linear Schedule
sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(optimiser_one, base_lr=0.005, max_lr=learning_rate, step_size_down=15, mode='triangular', verbose=False)
sched_linear_3 = torch.optim.lr_scheduler.LinearLR(optimiser_one, start_factor=0.005/learning_rate, end_factor=0.005/5, verbose=False)
scheduler = torch.optim.lr_scheduler.SequentialLR(optimiser_one, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])

# SGD optimiser 2
learning_rate = 0.001
optimiser_two = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Optimisers
optimisers = [optimiser_one, optimiser_two]

# Training loop
loss_epoch_train = {}
loss_epoch_val = {}
print('start Training: ')
for run, optimiser in enumerate(optimisers):
    for epoch in range(num_epochs[run]):  # loop over the dataset multiple times
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader
            
            running_loss = 0.0
            for i, data in enumerate(dataloader):
                percentageDone = int((i/ len(dataloader)) * 100)
                if percentageDone % int(len(dataloader) / 85) == 0:
                    print(f"\rProgress: {percentageDone}/{100}", end="", flush=True)
                    
                anchor, positive, negative = data
                anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
                optimiser.zero_grad()

                # Forward pass
                output1, output2, output3 = model(anchor, positive, negative)  # Pass two images as inputs
                loss = criterion(output1, output2, output3)
                
                if phase == 'train':
                    loss.backward()
                    optimiser.step()

                running_loss += loss.item()
                break
            print(f"\r", end="", flush=True)
            print(f'{epoch} {phase} Loss: {running_loss / len(dataloader)}')
            if phase == 'train':
                loss_epoch_train[run * num_epochs[0] + epoch] = running_loss / len(dataloader)
            else:
                loss_epoch_val[run * num_epochs[0] + epoch] = running_loss / len(dataloader)

print('Finished Training')
torch.save(model.state_dict(), 'TripleNet.pth')

plt.figure()
plt.plot(list(loss_epoch_train.keys()), list(loss_epoch_train.values()))
plt.plot(list(loss_epoch_val.keys()), list(loss_epoch_val.values()))
plt.savefig('running_loss_triplet_network.png')
