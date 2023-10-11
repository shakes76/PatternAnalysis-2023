import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
# import torchvision.datasets as dset
# import torchvision.transforms.v2 as transforms
# import torchvision.utils as vutils
# import numpy as np
import matplotlib.pyplot as plt

from modules import SiameseTwin, SiameseNeuralNet, SiameseMLP
from dataset import PairedDataset, load_data

TRAIN_PATH = '/home/groups/comp3710/ADNI/AD_NC/train/'
TEST_PATH = '/home/groups/comp3710/ADNI/AD_NC/test/'
RESULTS_PATH = "/home/Student/s4641725/COMP3710/project_results/"

# Loss Functions and Optimizers -----------------------------------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = (label) * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        loss = torch.mean(loss)

        return loss

def contrastive_loss(x1:torch.Tensor, x2:torch.Tensor, label:torch.Tensor, margin:float=1.0):
    
    difference = F.pairwise_distance(x1, x2)
    loss = (label * torch.pow(difference, 2) + 
            (1-label) * torch.max(torch.zeros_like(difference), margin-torch.pow(difference, 2)))
    loss = torch.mean(loss)
    return loss


# def weights_initialisation(model:nn.Module):
#     classname = model.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.01)
#         nn.init.normal_(model.bias.data)
#     elif classname.find('Linear') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.2)
        

# Training Loop ----------------------------------
def initialise_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Device: ", device)

    siamese_net = SiameseNeuralNet()
    siamese_net = siamese_net.to(device)
    print(siamese_net)

    criterion = ContrastiveLoss()
    optimiser = optim.Adam(siamese_net.parameters(), lr=1e-3, betas=(0.9, 0.999))
    return siamese_net, criterion, optimiser, device

def load_from_checkpoint(filename:str):
    pass

def save_checkpoint():
    pass

def train_siamese_one_epoch(model: nn.Module, 
                            criterion: nn.Module, 
                            optimiser: optim.Optimizer, 
                            device: torch.device,
                            train_loader: torch.utils.data.DataLoader):
    model.train()
    start = time.time()
    num_batches = len(train_loader)
    total_loss = 0.0
    for i, (x1, x2, label) in enumerate(train_loader, 0):
        # forward pass
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.to(device)

        optimiser.zero_grad()
        sameness, x1_features, x2_features = model(x1, x2)
        loss = criterion(x1_features, x2_features, label)

        # Backward and optimize
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0:
            print(f"Step [{i+1}/{len(train_loader)}] Loss: {loss.item()}")

        total_loss += loss.item()

    mean_loss = total_loss / num_batches
    end = time.time()
    elapsed = end - start

    return total_loss, mean_loss, elapsed

def eval_siamese_one_epoch(model: nn.Module,
                            criterion: nn.Module,
                            device: torch.device,
                            test_loader: torch.utils.data.DataLoader):
    model.eval()
    start = time.time()
    num_batches = len(test_loader)
    total_loss = 0.0
    with torch.no_grad(): # disables gradient calculation
        # correct = 0
        # total = 0
        for i, (x1, x2, labels) in enumerate(test_loader, 0):
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            sameness, x1_features, x2_features = model(x1, x2)

            loss = criterion(x1_features, x2_features, labels)

            if (i+1) % 100 == 0:
                print(f"Step [{i+1}/{len(test_loader)}] Loss: {loss.item()}")

            total_loss += loss.item()
        
        mean_loss = total_loss / num_batches
        
    end = time.time()
    elapsed = end - start

    return total_loss, mean_loss, elapsed

def train_classifier_one_epoch(model: nn.Module,
                                criterion: nn.Module,
                                optimiser: optim.Optimizer,
                                device: torch.device,
                                train_loader: torch.utils.data.DataLoader):
    model.train()
    start = time.time()
    num_batches = len(train_loader)
    total_loss = 0.0
    for i, (x, label) in enumerate(train_loader, 0):
        # forward pass
        x = x.to(device)
        label = label.to(device)

        optimiser.zero_grad()
        out = model(x)
        loss = criterion(out, label)

        # Backward and optimize
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0:
            print(f"Step [{i+1}/{len(train_loader)}] Loss: {loss.item()}")

        total_loss += loss.item()

    mean_loss = total_loss / num_batches
    end = time.time()
    elapsed = end - start

    return total_loss, mean_loss, elapsed

def eval_classifier_one_epoch(model: nn.Module,
                                criterion: nn.Module,
                                device: torch.device,
                                test_loader: torch.utils.data.DataLoader):
    model.eval()
    start = time.time()
    num_batches = len(test_loader)
    total_loss = 0.0
    with torch.no_grad(): # disables gradient calculation
        correct = 0
        total = 0
        total_loss = 0.0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.round(outputs)
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy: {} %'.format(100 * correct / total))
        mean_loss = total_loss / num_batches
    end = time.time()
    elapsed = end - start

    return total_loss, mean_loss, elapsed

def test_loss():
    cont_loss = ContrastiveLoss()
    input1 = torch.rand(2, 4096)
    input2 = torch.rand(2, 4096)
    
    labels = torch.Tensor([1,0])
    old_loss = contrastive_loss(input1, input2, labels)
    new_loss = cont_loss(input1, input2, labels)
    print(old_loss)
    print(new_loss)

if __name__ == "__main__":
    starting_epoch = 0
    num_epochs = 10
    random_seed = 69

    train_loader = load_data(training=True, Siamese=True, random_seed=random_seed)
    test_loader = load_data(training=False, Siamese=True, random_seed=random_seed)
    siamese_net, criterion, optimiser, device = initialise_training()

    training_losses = []
    eval_losses = []

    print('starting training and validation loop for Siamese backbone')
    start = time.time()

    for epoch in range(starting_epoch, num_epochs):
        print(f'Training Epoch {epoch+1}')
        train_loss, avg_train_loss, elapsed = train_siamese_one_epoch(siamese_net, criterion, optimiser, device, train_loader)
        training_losses.append(avg_train_loss)
        print(f'Training Epoch {epoch+1} took {elapsed:.1f} seconds. Total loss: {train_loss:.4f}. Average loss: {avg_train_loss:.4f}')

        print(f'Validating Epoch {epoch+1}')
        eval_loss, avg_eval_loss, elapsed = eval_siamese_one_epoch(siamese_net, criterion, device, test_loader)
        eval_losses.append(avg_eval_loss)
        print(f'Validating Epoch {epoch+1} took {elapsed:.1f} seconds. Total loss: {eval_loss:.4f}. Average loss: {avg_eval_loss:.4f}')

    end = time.time()
    elapsed = end - start
    print("Training and Validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    plt.figure(figsize=(10,5))
    plt.title("Training and Evaluation Loss During Training")
    plt.plot(training_losses, label="Train")
    plt.plot(eval_losses, label="Eval")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(RESULTS_PATH + f"Siamese_train_and_eval_loss_after_{num_epochs}_epochs.png")
 
    # classifier training
    starting_epoch = 0
    train_loader = load_data(training=True, Siamese=False, random_seed=random_seed)
    test_loader = load_data(training=False, Siamese=False, random_seed=random_seed)

    backbone = siamese_net.get_backbone()
    classifier = SiameseMLP(backbone)
    criterion = nn.BCELoss()
    optimiser = optim.Adam(classifier.parameters(), lr=1e-3, betas=(0.9, 0.999))

    training_losses = []
    eval_losses = []
    print('starting training and validation loop for Classifier')
    start = time.time()

    for epoch in range(starting_epoch, num_epochs):
        print(f'Training Epoch {epoch+1}')
        train_loss, avg_train_loss, elapsed = train_classifier_one_epoch(classifier, criterion, optimiser, device, train_loader)
        training_losses.append(avg_train_loss)
        print(f'Training Epoch {epoch+1} took {elapsed:.1f} seconds. Total loss: {train_loss:.4f}. Average loss: {avg_train_loss:.4f}')

        print(f'Validating Epoch {epoch+1}')
        eval_loss, avg_eval_loss, elapsed = eval_classifier_one_epoch(classifier, criterion, device, test_loader)
        eval_losses.append(avg_eval_loss)
        print(f'Validating Epoch {epoch+1} took {elapsed:.1f} seconds. Total loss: {eval_loss:.4f}. Average loss: {avg_eval_loss:.4f}')

    end = time.time()
    elapsed = end - start
    print("Training and Validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    plt.figure(figsize=(10,5))
    plt.title("Training and Evaluation Loss During Training")
    plt.plot(training_losses, label="Train")
    plt.plot(eval_losses, label="Eval")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(RESULTS_PATH + f"Classifier_train_and_eval_loss_after_{num_epochs}_epochs.png")


    print('END')


    # load_from_checkpoint()
    # train_and_eval()
    # test_loss()

#
# Deprecated
#
def train_and_eval():
    starting_epoch = 0
    num_epochs = 5
    random_seed = 69

    train_loader = load_data(training=True, Siamese=True, random_seed=random_seed)
    siamese_net, criterion, optimiser, device = initialise_training()

    total_step = len(train_loader)

    # scheduler = optim.lr_scheduler.ExponentialLR(optimiser, 0.99)

    training_losses = []

    siamese_net.train()
    print("> Training")
    start = time.time() #time generation
    for epoch in range(starting_epoch, num_epochs):
    # For each batch in the dataloader
        for i, (x1, x2, label) in enumerate(train_loader, 0):
            # forward pass
            x1 = x1.to(device)
            x2 = x2.to(device)
            label = label.to(device)

            optimiser.zero_grad()
            sameness, x1_features, x2_features = siamese_net(x1, x2)
            loss = criterion(x1_features, x2_features, label)

            # Backward and optimize
            loss.backward()
            optimiser.step()

            if (i+1) % 100 == 0:
                print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
            training_losses.append(loss.item()) 
        # stepping LR scheduler every epoch rather than every batch
        # scheduler.step()
    end = time.time()
    elapsed = end - start
    print("Training took " + str(elapsed) + " secs of " + str(elapsed/60) + " mins in total")

    # --------------
    # Test the model
    print("> Testing")
    start = time.time() #time generation
    siamese_net.eval()
    test_loader = load_data(training=False, Siamese=True, random_seed=random_seed)

    eval_losses = []

    with torch.no_grad(): # disables gradient calculation
        correct = 0
        total = 0
        for x1, x2, labels in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            sameness, x1_features, x2_features = siamese_net(x1, x2)
            predicted = torch.round(sameness)

            loss = criterion(x1_features, x2_features, labels)
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            eval_losses.append(loss.item())
            
        print('Test Accuracy: {} %'.format(100 * correct / total))
    
    end = time.time()
    elapsed = end - start
    print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    print('END')

    plt.figure(figsize=(10,5))
    plt.title("Training and Evaluation Loss During Training")
    plt.plot(training_losses, label="Train")
    plt.plot(eval_losses, label="Eval")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(RESULTS_PATH + f"train_and_eval_loss_after_{num_epochs}_epochs.png")