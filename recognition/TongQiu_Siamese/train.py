import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from utils import Config
from torch.utils.tensorboard import SummaryWriter


def main(model, train_loader, val_loader, criterion, optimizer, epochs):
    print('---------Train on: ' + Config.DEVICE + '----------')

    # Create model
    model = model.to(Config.DEVICE)
    best_score = 0
    writer = SummaryWriter(log_dir=Config.LOG_DIR)  # for TensorBoard

    for epoch in range(epochs):

        # train
        train_batch_loss, train_batch_acc = train(model, train_loader, optimizer, criterion, epoch, epochs)
        # validate
        val_batch_loss, val_batch_acc = validate(model, val_loader, criterion, epoch, epochs)

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
        writer.add_scalar("Training Loss", train_batch_loss.item(), epoch)
        writer.add_scalar("Training Score", train_batch_acc.item(), epoch)
        writer.add_scalar("Validation Loss", val_batch_loss.item(), epoch)
        writer.add_scalar("Validation Score", val_batch_acc.item(), epoch)

    writer.close()


def train(model, train_loader, optimizer, criterion, epoch, epochs):
    model.train()
    train_loss_lis = np.array([])
    correct_predictions = 0
    total_samples = 0
    negative_pairs_below_margin = 0  # Count of negative pairs with distances below the margin
    total_negative_pairs = 0

    for batch in tqdm(train_loader):
        vols_1, vols_2, labels = batch['volume1'], batch['volume2'], batch['label']
        vols_1, vols_2, labels = vols_1.to(Config.DEVICE), vols_2.to(Config.DEVICE), labels.to(Config.DEVICE)

        optimizer.zero_grad()
        embedding_1, embedding_2 = model(vols_1, vols_2)
        loss = criterion(embedding_1, embedding_2, labels)
        loss.backward()
        optimizer.step()

        # Record the batch loss
        train_loss_lis = np.append(train_loss_lis, loss.item())

        # Compute pairwise distance using F.pairwise_distance
        dists = F.pairwise_distance(embedding_1, embedding_2)

        # Use the criterion's margin as the threshold for predictions
        threshold = criterion.margin
        predictions = (dists < threshold).float()
        correct_predictions += (predictions == labels.float()).sum().item()
        total_samples += labels.size(0)

        # Update the count of negative pairs below the margin
        negative_pair_mask = (labels == 0).float()  # 0 for negative pair
        total_negative_pairs += negative_pair_mask.sum().item()
        negative_dists_below_margin = (dists < criterion.margin).float() * negative_pair_mask
        negative_pairs_below_margin += negative_dists_below_margin.sum().item()

    train_loss = sum(train_loss_lis) / len(train_loss_lis)
    accuracy = 100.0 * correct_predictions / total_samples

    # Adjust the margin if too many negative pairs are below it
    proportion_negative_below_margin = negative_pairs_below_margin / (total_negative_pairs + 1e-10)
    if proportion_negative_below_margin > 0.3:  # Example threshold, adjust as needed
        criterion.margin *= 0.95  # Reduce the margin by 5%

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{epochs:03d} ] margin = {criterion.margin}, acc = {accuracy:.5f}, loss = {train_loss:.5f}")
    return train_loss, accuracy


def validate(model, val_loader, criterion, epoch, epochs):
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(val_loader):
            vols_1, vols_2, labels = batch['volume1'], batch['volume2'], batch['label']
            vols_1, vols_2, labels = vols_1.to(Config.DEVICE), vols_2.to(Config.DEVICE), labels.to(Config.DEVICE)
            embedding_1, embedding_2 = model(vols_1, vols_2)
            loss = criterion(embedding_1, embedding_2, labels)

            total_loss += loss.item()

            # Compute pairwise distance
            dists = F.pairwise_distance(embedding_1, embedding_2)
            threshold = criterion.margin
            predictions = (dists < threshold).float()
            correct_predictions += (predictions == labels.float()).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / len(val_loader)
    val_acc = 100.0 * correct_predictions / total_samples

    # Print the information.
    print(f"[ Validation | {epoch + 1:03d}/{epochs:03d} ] margin = {criterion.margin:.5f}, acc = {val_acc:.5f}, loss = {average_loss:.5f}")

    return average_loss, val_acc

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin  # margin specifies how far apart the embeddings of dissimilar pairs should be

    def forward(self, logits1, logits2, label):
        euclidean_distance = F.pairwise_distance(logits1, logits2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive
