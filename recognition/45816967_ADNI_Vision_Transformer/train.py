import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import *
from vit import *
import torch.optim as optim
from tqdm import tqdm

from torch.utils.data import DataLoader


def train(model, train_loader, val_loader, criterion, n_epochs, lr):
  # Define and optimizer
  optimizer = optim.Adam(model.parameters(), lr=lr)

  # Train the model
  for epoch in range(n_epochs):
    # Train the model for one epoch
    model.train()
    train_loss = 0.0
    for imgs, labels in tqdm(train_loader):
      optimizer.zero_grad()
      out = model(imgs)
      loss = criterion(out, labels)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * imgs.size(0)
    train_loss /= len(train_loader.dataset)

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    with torch.no_grad():
      for imgs, labels in tqdm(val_loader):
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        val_acc += torch.sum(preds == labels.data)
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)

    # Print the loss and accuracy for this epoch
    print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'
        .format(epoch+1, n_epochs, train_loss, val_loss, val_acc))
  
  

# train_imgs, test_imgs = load_adni_images(verbose = False)
# # print(train_imgs[0][0])

# # plt.imshow(train_imgs[0][0][0], cmap="gray")
# # plt.show()

# train_set = ADNIDataset(train_imgs[0], transform=standardTransform)
# val_set = ADNIDataset(train_imgs[1], transform=standardTransform)
# test_set = ADNIDataset(test_imgs[0], transform=standardTransform)

# train_loader = DataLoader(train_set, batch_size=2)
# val_loader = DataLoader(val_set, batch_size=2)
# test_loader = DataLoader(val_set, batch_size=2)
# # nextimg = next(iter(dl))
# # print(nextimg)

# model = ViT(image_size=128, patch_size=32, num_classes=2, dim=1024, depth=6, heads=16, mlp_dim=2048, dropout=0.1)

# train(model, train_loader, val_loader, nn.CrossEntropyLoss(), n_epochs=10, lr=0.001)