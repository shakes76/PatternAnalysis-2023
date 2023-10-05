import dataset
from modules import *
import torch.optim as optim


#put in training loop
epochs = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ADNI_Transformer(depth=3)
model.to(device=device)

dataset = ds.ADNI_Dataset()
train_loader = dataset.get_train_loader()
test_loader = dataset.get_test_loader()
optimizer = optim.Adam(model.parameters(), 1e-5)
criterion = nn.BCELoss()

batch_losses = []


model.train()
for epoch in range(epochs):
    batch_loss = 0

    for j, (images, labels) in  enumerate(train_loader):
        if images.size(0) == 32:
            optimizer.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            loss = criterion(outputs.squeeze().to(torch.float), labels.to(torch.float))
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
    
    batch_losses.append(batch_loss)
    print(batch_loss)
