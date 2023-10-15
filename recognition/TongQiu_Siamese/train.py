import numpy as np
import torch
from tqdm.auto import tqdm
from utils import Config
from torch.utils.tensorboard import SummaryWriter

def main(model, train_loader, val_loader, criterion, optimizer, epochs):
    print('---------Train on: ' + Config.DEVICE + '----------')

    # Create model
    model = model.to(Config.DEVICE)
    best_score = 0

    # Create a SummaryWriter for TensorBoard
    writer = SummaryWriter(log_dir=Config.LOG_DIR)

    # For epoch in range(args.epochs):
    for epoch in range(epochs):
        # train
        train_batch_loss, train_batch_acc = train(model, train_loader, optimizer, criterion, epoch, epochs)

        # validate
        val_batch_acc = validate(model, val_loader, epoch, epochs)

        # Save model
        if val_batch_acc > best_score:
            print(f"model improved: score {best_score:.5f} --> {val_batch_acc:.5f}")
            best_score = val_batch_acc
            # Save the best weights if the score is improved
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_batch_loss,
                'train_acc': train_batch_acc,
                'val_acc': val_batch_acc
            }, Config.MODEL_DIR)
        else:
            print(f"no improvement: score {best_score:.5f} --> {val_batch_acc:.5f}")

        # Write loss and score to TensorBoard
        writer.add_scalar("Training Loss", train_batch_loss.item(), epoch)
        writer.add_scalar("Validation Score", train_batch_acc.item(), epoch)

def train(model, train_loader, optimizer, criterion, epoch, epochs):
    model.train()
    train_loss_lis = np.array([])
    train_acc_lis = np.array([])
    for batch in tqdm(train_loader):
        vols_1, vols_2, labels = batch['volume1'], batch['volume2'], batch['label']
        vols_1, vols_2, labels = vols_1.to(Config.DEVICE), vols_2.to(Config.DEVICE), labels.to(Config.DEVICE)
        # labels = torch.nn.functional.one_hot(labels).long().to(device)
        logits = model(vols_1, vols_2)
        loss = criterion(logits, labels)

        # Compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate the batch acc
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        # Record the batch loss and accuracy.
        train_loss_lis = np.append(train_loss_lis, loss.item())
        train_acc_lis = np.append(train_acc_lis, acc.cpu())

    train_loss = sum(train_loss_lis) / len(train_loss_lis)
    train_acc = sum(train_acc_lis) / len(train_acc_lis)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    return train_loss, train_acc


def validate(model, val_loader, epoch, epochs):
    model.eval()
    val_acc_lis = np.array([])
    for batch in tqdm(val_loader):
        vols_1, vols_2, labels = batch['volume1'], batch['volume2'], batch['label']

        with torch.no_grad():
            vols_1, vols_2, labels = vols_1.to(Config.DEVICE), vols_2.to(Config.DEVICE), labels.to(Config.DEVICE)
            logits = model(vols_1, vols_2)

            # Calculate the batch acc
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the batch loss and accuracy.
            val_acc_lis = np.append(val_acc_lis, acc.cpu())

    val_acc = sum(val_acc_lis) / len(val_acc_lis)

    # Print the information.
    print(f"[ Validation | {epoch + 1:03d}/{epochs:03d} ]  acc = {val_acc:.5f}")
    return val_acc

"""
# Test main(), train(), validate()

from modules import Baseline
from dataset import ContrastiveDataset
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
if __name__ == '__main__':

    model= Baseline()

    full_train_dataset = ContrastiveDataset(Config.TRAIN_DIR)
    # Split the full training dataset into train and validation sets
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    dataset_tr, dataset_val = random_split(full_train_dataset, [train_size, val_size])

    dataloader_tr = DataLoader(
        dataset=dataset_tr,
        shuffle=True,
        batch_size=3,
        num_workers=1,
        drop_last=True
    )
    dataloader_val = DataLoader(
        dataset=dataset_val,
        shuffle=True,
        batch_size=3,
        num_workers=1,
        drop_last=True
    )

    criterion = nn.CrossEntropyLoss()

    lr = 0.005
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 5

    main(model, dataloader_tr, dataloader_val, criterion, optimizer, epochs)
"""
