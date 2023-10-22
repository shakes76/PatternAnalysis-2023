import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as tf
from tqdm.auto import tqdm
from utils import Config
from torch.utils.tensorboard import SummaryWriter
import random
import os
from modules import Embedding_Baseline, SiameseContrastive, SiameseTriplet, ClassificationNet
from dataset import ContrastiveDataset, discover_directory, patient_level_split, TripletDataset
from torch.utils.data import DataLoader
import argparse

"""
Trining process for Contrastive loss
"""


def main_contrastive(model, train_loader, val_loader, criterion, optimizer, epochs):
    print('---------Siamese(Contrastive) Train on: ' + Config.DEVICE + '----------')

    # Create model
    model = model.to(Config.DEVICE)
    best_loss = float('inf')

    # Lists for storing metrics
    train_losses = []
    val_losses = []
    early_stopping_counter = 0

    for epoch in range(epochs):

        # train
        train_batch_loss = train_contrastive(model, train_loader, optimizer, criterion, epoch, epochs)
        train_losses.append(train_batch_loss)

        # validate
        val_batch_loss = validate_contrastive(model, val_loader, criterion, epoch, epochs)
        val_losses.append(val_batch_loss)

        if val_batch_loss < best_loss:
            print(f"model improved: score {best_loss:.5f} --> {val_batch_loss:.5f}")
            best_loss = val_batch_loss
            # Save the best weights if the score is improved
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.embedding_net.state_dict(),
                'train_loss': train_batch_loss,
            }, Config.MODEL_DIR)
        else:
            early_stopping_counter += 1
            print(f"no improvement: loss {best_loss:.5f} --> {val_batch_loss:.5f}")

        if early_stopping_counter > 5:
            print("Early stopping!")
            break

    # Convert lists to numpy arrays
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # Save the arrays
    np.save(os.path.join(Config.LOG_DIR_CONTRASTIVE, 'contrastive/train_losses.npy'), train_losses)
    np.save(os.path.join(Config.LOG_DIR_CONTRASTIVE, 'contrastive/val_losses.npy'), val_losses)


def train_contrastive(model, train_loader, optimizer, criterion, epoch, epochs):
    model.train()
    train_loss_lis = np.array([])
    negative_pairs_below_margin = 0  # Count of negative pairs with distances below the margin
    total_negative_pairs = 0

    for batch in tqdm(train_loader):
        img_1, img_2, labels = batch
        img_1, img_2, labels = img_1.to(Config.DEVICE), img_2.to(Config.DEVICE), labels.to(Config.DEVICE)

        optimizer.zero_grad()
        embedding_1, embedding_2 = model(img_1, img_2)
        loss = criterion(embedding_1, embedding_2, labels)
        loss.backward()
        optimizer.step()

        # Record the batch loss
        train_loss_lis = np.append(train_loss_lis, loss.item())

        # Update the count of negative pairs below the margin
        negative_pair_mask = (labels == 0).float()  # 0 for negative pair
        total_negative_pairs += negative_pair_mask.sum().item()
        dists = model.euclidean_distance(embedding_1, embedding_2)
        negative_dists_below_margin = (dists < criterion.margin).float() * negative_pair_mask.squeeze()
        negative_pairs_below_margin += negative_dists_below_margin.sum().item()

    # calculate training loss
    train_loss = sum(train_loss_lis) / len(train_loss_lis)

    # Adjust the margin if too many negative pairs are below it
    proportion_negative_below_margin = negative_pairs_below_margin / (total_negative_pairs + 1e-10)
    if proportion_negative_below_margin > 0.3:  # Example threshold, adjust as needed
        criterion.margin *= 0.9  # Reduce the margin by 10%
        print('proportion_negative_below_margin', proportion_negative_below_margin)

    # Print the information.
    print(
        f"[ Train | {epoch + 1:03d}/{epochs:03d} ] margin = {criterion.margin}, loss = {train_loss:.5f}")
    return train_loss


def validate_contrastive(model, val_loader, criterion, epoch, epochs):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            img_1, img_2, labels = batch
            img_1, img_2, labels = img_1.to(Config.DEVICE), img_2.to(Config.DEVICE), labels.to(Config.DEVICE)
            embedding_1, embedding_2 = model(img_1, img_2)
            loss = criterion(embedding_1, embedding_2, labels)

            total_loss += loss.item()

    average_loss = total_loss / len(val_loader)

    # Print the information.
    print(
        f"[ Validation | {epoch + 1:03d}/{epochs:03d} ] margin = {criterion.margin:.5f}, loss = {average_loss:.5f}")

    return average_loss


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # margin specifies how far apart the embeddings of dissimilar pairs should be

    def forward(self, embedding1, embedding2, label):
        euclidean_distance = F.pairwise_distance(embedding1, embedding2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive


"""
Training process for Triplet loss
"""


def main_triplet(model, train_loader, val_loader, criterion, optimizer, epochs):
    print('---------Train on: ' + Config.DEVICE + '----------')

    # Create model
    model = model.to(Config.DEVICE)
    best_score = 0
    writer = SummaryWriter(log_dir=Config.LOG_DIR_TRIPLET)  # for TensorBoard

    for epoch in range(epochs):

        # train
        train_batch_loss, train_batch_acc = train_triplet(model, train_loader, optimizer, criterion, epoch, epochs)
        # validate
        val_batch_loss, val_batch_acc = validate_triplet(model, val_loader, criterion, epoch, epochs)

        if val_batch_acc > best_score:
            print(f"model improved: score {best_score:.5f} --> {val_batch_acc:.5f}")
            best_score = val_batch_acc
            # Save the best weights if the score is improved
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_batch_loss,
                'val_acc': val_batch_acc
            }, Config.MODEL_DIR)
        else:
            print(f"no improvement: score {best_score:.5f} --> {val_batch_acc:.5f}")

        # Write loss and score to TensorBoard
        writer.add_scalar("Training Loss", train_batch_loss, epoch)
        writer.add_scalar("Training Score", train_batch_acc, epoch)
        writer.add_scalar("Validation Loss", val_batch_loss, epoch)
        writer.add_scalar("Validation Score", val_batch_acc, epoch)

    writer.close()


def train_triplet(model, train_loader, optimizer, criterion, epoch, epochs):
    model.train()
    train_loss_lis = np.array([])

    for batch in tqdm(train_loader):
        anchor, positive, negative, labels = batch
        anchor, positive, negative, labels = anchor.to(Config.DEVICE), positive.to(Config.DEVICE), \
            negative.to(Config.DEVICE), labels.to(Config.DEVICE)

        optimizer.zero_grad()
        embedding_a, embedding_p, embedding_n = model(anchor, positive, negative)
        loss = criterion(embedding_a, embedding_p, embedding_n)
        loss.backward()
        optimizer.step()

        # Record the batch loss
        train_loss_lis = np.append(train_loss_lis, loss.item())

    train_loss = sum(train_loss_lis) / len(train_loss_lis)

    # Print the information.
    print(
        f"[ Train | {epoch + 1:03d}/{epochs:03d} ] loss = {train_loss:.5f}")
    return train_loss


def validate_triplet(model, val_loader, criterion, epoch, epochs):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            anchor, positive, negative, labels = batch
            anchor, positive, negative, labels = anchor.to(Config.DEVICE), positive.to(Config.DEVICE), \
                negative.to(Config.DEVICE), labels.to(Config.DEVICE)
            embedding_a, embedding_p, embedding_n = model(anchor, positive, negative)
            loss = criterion(embedding_a, embedding_p, embedding_n)

            total_loss += loss.item()

            # Compute accuracy
            dists_p = F.pairwise_distance(embedding_a, embedding_p)
            dists_n = F.pairwise_distance(embedding_a, embedding_n)
            pred_diff = (dists_p < dists_n).float()
            predictions = pred_diff * labels.squeeze() + (1 - pred_diff) * (1 - labels.squeeze())
            correct_predictions += (predictions == labels.squeeze().float()).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(val_loader)
    val_acc = correct_predictions / total_samples

    # Print the information.
    print(
        f"[ Validation | {epoch + 1:03d}/{epochs:03d} ] acc = {val_acc:.5f}, loss = {average_loss:.5f}")

    return average_loss, val_acc

"""
# train with Triplet loss
if __name__ == '__main__':
    random.seed(2023)
    model = Baseline_Triplet()

    full_train_data = discover_directory(Config.TRAIN_DIR)
    train_data, val_data = patient_level_split(full_train_data)  # patient-level split

    tr_transform = tf.Compose([
        tf.Normalize((0.1160,), (0.2261,)),
        tf.RandomRotation(10)
    ])
    val_transform = tf.Compose([
        tf.Normalize((0.1160,), (0.2261,)),
        tf.RandomRotation(10)
    ])

    train_dataset = TripletDataset(train_data, transform=tr_transform)
    val_dataset = TripletDataset(val_data, transform=val_transform)

    dataloader_tr = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=8,
        num_workers=1,
        drop_last=True
    )
    dataloader_val = DataLoader(
        dataset=val_dataset,
        shuffle=True,
        batch_size=8,
        num_workers=1,
        drop_last=True
    )

    criterion = torch.nn.TripletMarginLoss()

    lr = 0.005
    weight_decay = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 50

    main_triplet(model, dataloader_tr, dataloader_val, criterion, optimizer, epochs)
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script for Contrastive/Triplet network')

    # model
    parser.add_argument('-m', '--model', default='Contrastive', type=str,
                        help='model to train (Contrastive/Triplet/Classification')

    # seed
    parser.add_argument('-sd', '--seed', default=2023, type=int, help='Seed for initializing training.')

    # data loading
    parser.add_argument('-tp', '--train_path', default=Config.TRAIN_DIR, help='Path to the training dataset.')
    parser.add_argument('-wn', '--workers', default=1, type=int, help='Number of workers for data loading.')
    parser.add_argument('-bs', '--batch_size', default=8, type=int, help='Batch size for training.')

    # training parameters
    parser.add_argument('-ep', '--epochs', default=50, type=int, help='Number of epochs for training.')
    parser.add_argument('-lr', '--learning_rate', default=0.005, type=float, help='Initial learning rate.')
    parser.add_argument('-wd', '--weight_decay', default=1e-5, type=float, help='Weight decay for the optimizer.')

    args = parser.parse_args()

    # seed
    random.seed(args.seed)

    # get data
    full_train_data = discover_directory(args.train_path)
    train_data, val_data = patient_level_split(full_train_data)  # patient-level split

    tr_transform = tf.Compose([
        tf.Normalize((0.1160,), (0.2261,)),
        tf.RandomRotation(10)
    ])
    val_transform = tf.Compose([
        tf.Normalize((0.1160,), (0.2261,)),
        tf.RandomRotation(10)
    ])

    lr = args.learning_rate
    weight_decay = args.weight_decay
    epochs = args.epochs

    # pretraining Siamese
    if args.model == 'Contrastive':
        embedding_net = Embedding_Baseline()
        model = SiameseContrastive(embedding_net)

        train_dataset = ContrastiveDataset(train_data, transform=tr_transform)
        val_dataset = ContrastiveDataset(val_data, transform=val_transform)

        dataloader_tr = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True
        )
        dataloader_val = DataLoader(
            dataset=val_dataset,
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
            drop_last=True
        )

        criterion = ContrastiveLoss(margin=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        main_contrastive(model, dataloader_tr, dataloader_val, criterion, optimizer, epochs)

    else:
        print('model type not included, try Contrastive/Triplet.')





