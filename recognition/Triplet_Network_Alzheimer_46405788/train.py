from dataset import get_classification_dataloader, get_triplet_train_loader, get_triplet_test_loader
from modules import TripletLoss, TripletNet, TripletNetClassifier
import torch
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

print('TripletNetwork')

batch_size = 32
num_epochs = [35, 100]

train_folder = 'AD_NC/train'
test_folder = 'AD_NC/test'

train_loader = get_triplet_train_loader(train_folder, batch_size)
test_loader = get_triplet_test_loader(test_folder, batch_size)
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
    for epoch in range(num_epochs[run]):
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
                output1, output2, output3 = model(anchor, positive, negative)
                loss = criterion(output1, output2, output3)
                
                if phase == 'train':
                    loss.backward()
                    optimiser.step()

                running_loss += loss.item()
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

print('Classifier')
tripleNet = model

def extract_features(data):
    with torch.no_grad():
        embeddings = tripleNet.forward_one(data)
        return embeddings

batch_size = 32

data_folder = 'AD_NC'

train_loader, test_loader = get_classification_dataloader(data_folder, batch_size)

# Load the pre-trained Siamese network
tripleClassifier = TripletNetClassifier().to(device)

# Loss function 
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(tripleClassifier.parameters(), lr=0.001)

t_loss = {}
v_loss = {}
num_epochs = 100

# Train the classifier using labeled data
print('start Training: ')
for epoch in range(num_epochs):
    for phase in ['train', 'val']:
        if phase == 'train':
            tripleClassifier.train()
            dataloader = train_loader
        else:
            tripleClassifier.eval()
            dataloader = test_loader
        
        running_loss = 0.0
        for i, data in enumerate(dataloader):

            percentageDone = int((i/ len(dataloader)) * 100)
            if percentageDone % int(len(dataloader) / 200) == 0:
                print(f"\rProgress: {percentageDone}/{100}", end="", flush=True)

            input, label = data
            input, label = input.to(device), label.to(device)
            features = extract_features(input)
            optimizer.zero_grad()
            outputs = tripleClassifier(features)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"\r", end="", flush=True)
        print(f'{epoch} {phase} Loss: {running_loss / len(dataloader)}')
        if phase == 'train':
            t_loss[epoch] = running_loss / len(dataloader)
        else:
            v_loss[epoch] = running_loss / len(dataloader)

torch.save(tripleClassifier.state_dict(), 'TripleNetClassifier.pth')

plt.figure()
plt.plot(list(t_loss.keys()), list(t_loss.values()))
plt.plot(list(v_loss.keys()), list(v_loss.values()))
plt.savefig('running_loss_triplet_classifier.png')

print('Finished Training')