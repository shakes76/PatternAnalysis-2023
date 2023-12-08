'''
contains the source code for training, validating, testing and saving the model.
The model is be imported from “modules.py” and the data loader is imported from “dataset.py”. 
Make sure to plot the losses and metrics during training.
'''

from dataset import get_alzheimer_dataloader
from modules import ViT
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

class_names = ["AD", "NC"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, train_loader, val_loader, criterion, optimizer, device, train_interval, val_interval, batch_size):
    model.train()
    total_loss = 0.0
    total_correct = 0
    batch_counter = 0

    # these will keep track of loss and correct counts for the previous 'print_interval' batches
    interval_loss = 0.0
    interval_correct = 0

    # training loop for 1 epoch
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device).long()

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # update the running totals
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total_correct += predicted.eq(labels).sum().item()

        # update the interval totals
        interval_loss += loss.item() * inputs.size(0)
        interval_correct += predicted.eq(labels).sum().item()

        # every n batches print the average loss and accuracy for the previous n batches
        batch_counter += 1
        if train_interval != 0 and batch_counter == train_interval:
            avg_interval_loss = interval_loss / (batch_size * train_interval)
            avg_interval_accuracy = interval_correct / (batch_size * train_interval)
            print(f"After {i+1} batches, Train Average Loss (last {train_interval} batches): {avg_interval_loss:.4f}, Average Accuracy (last {train_interval} batches): {avg_interval_accuracy:.4f}")
            # reset interval counts for the next set of 'print_interval' batches
            interval_loss = 0.0
            interval_correct = 0
            batch_counter = 0

        if val_interval != 0 and i != 0 and i % val_interval == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f"After {i+1} batches, Validation Average Loss: {val_loss:.4f}, Validation Average Accuracy: {val_acc:.4f}")


    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / len(train_loader.dataset)
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    """Evaluate the model without updating weights."""
    model.eval()
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device).long()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total_correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = total_correct / len(loader.dataset)
    return avg_loss, accuracy


def plot_training(train_epoch_accuracies, val_epoch_accuracies, test_accuracy, num_epochs):

    plt.figure(figsize=(15, 10))

    epoch_train_x_values = range(1, num_epochs+1)

    # plotting epoch accuracies
    plt.plot(epoch_train_x_values, train_epoch_accuracies, label='Train Epoch Acc', color='blue')
    plt.plot(epoch_train_x_values , val_epoch_accuracies, label='Val Epoch Acc', color='red')

    # plotting max validation point
    plt.scatter(val_epoch_accuracies.index(max(val_epoch_accuracies))+1, max(val_epoch_accuracies), s=100, color='blue', label=f'Best Val Acc: {max(val_epoch_accuracies)*100:.0f}%')

    plt.axhline(y=test_accuracy, color='green', linestyle='-', alpha=0.5, label=f'Test Accuracy: {test_accuracy*100:.0f}%')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.show()
    plt.savefig('training_plot.png', bbox_inches='tight')

def main():

    parser = argparse.ArgumentParser()
    
    # add arguments for hyperparameters
    parser.add_argument('--lr', type=float, default=0.00001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.003, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--model_size', type=str, default='base', help='base, large, huge')
    parser.add_argument('--patch_size', type=int, default=16, help='Number of pixels in a patch')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout in ViT')
    parser.add_argument('--img_size', type=int, default=224, help='Size to resize image to')
    parser.add_argument('--path', type=str, default="/home/groups/comp3710/ADNI/AD_NC", help='Path to the dataset')
    parser.add_argument('--plot', type=bool, default=False, help='Include plot after training')
    parser.add_argument('--train_interval', type=int, default=0, help='Number of training batches before training stats are printed.')
    parser.add_argument('--val_interval', type=int, default=0, help='Number of training batches before validation stats are printed')

    # parse the arguments
    args = parser.parse_args()

    # now use the arguments
    lr = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.num_epochs
    model_size = args.model_size
    patch_size = args.patch_size
    batch_size = args.batch_size
    dropout = args.dropout
    img_size = args.img_size
    path = args.path

    # get loaders from dataset.py 
    train_loader, val_loader, test_loader = get_alzheimer_dataloader(batch_size=batch_size, img_size=img_size, path=path)

    # set for training and evaluating
    model = ViT(model_type=model_size, img_size=img_size, patch_size=patch_size, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)


    # initialize best validation accuracy at 0
    best_val_acc = 0.0

    train_epoch_accuracies = []
    val_epoch_accuracies = []


    # loop through the number of epochs
    for epoch in range(num_epochs):
        # train model
        train_loss, train_acc= train_epoch(model, train_loader, val_loader,
                                           criterion, optimizer, device, args.train_interval,
                                             args.val_interval, batch_size)
        
        # evaluate model
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train: Loss {train_loss:.4f}, Accuracy {train_acc:.4f}")
        print(f"Validation: Loss {val_loss:.4f}, Accuracy {val_acc:.4f}")

        # save model if it performs better on the validation set
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'vit-valacc{best_val_acc*100:.2f}.pth')
            print(f"New best validation accuracy ({best_val_acc*100:.2f}%) achieved! Model saved.")

        # save epoch accuracies for plotting
        train_epoch_accuracies.append(train_acc)
        val_epoch_accuracies.append(val_acc)

    # after all epochs, test the model performance on test data
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test: Loss {test_loss:.4f}, Accuracy {test_acc:.4f}")

    # plot training curve if allowed by user
    if args.plot:
        plot_training(train_epoch_accuracies, val_epoch_accuracies, test_acc, num_epochs)

if __name__ == "__main__":
    main()