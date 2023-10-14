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

    writer = SummaryWriter(log_dir=Config.LOG_DIR)  # Create a SummaryWriter for TensorBoard

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

def train(model, train_loader, optimizer, criterion, epoch, epochs):
    model.train()
    train_loss_lis = np.array([])
    train_acc_lis = np.array([])
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
        # labels = torch.nn.functional.one_hot(labels).long().to(device)
        logits = model(imgs)
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
        imgs, labels = batch

        with torch.no_grad():
            imgs, labels = imgs.to(Config.DEVICE), labels.to(Config.DEVICE)
            logits = model(imgs)

            # Calculate the batch acc
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the batch loss and accuracy.
            val_acc_lis = np.append(val_acc_lis, acc.cpu())

    val_acc = sum(val_acc_lis) / len(val_acc_lis)

    # Print the information.
    print(f"[ Validation | {epoch + 1:03d}/{epochs:03d} ]  acc = {val_acc:.5f}")
    return val_acc

