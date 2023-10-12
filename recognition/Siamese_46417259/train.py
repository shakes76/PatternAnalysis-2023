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
def initialise_Siamese_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Device: ", device)

    siamese_net = SiameseNeuralNet()
    siamese_net = siamese_net.to(device)
    print(siamese_net)

    criterion = ContrastiveLoss()
    optimiser = optim.Adam(siamese_net.parameters(), lr=1e-3, betas=(0.9, 0.999))
    return siamese_net, criterion, optimiser, device

def initialise_classifier_training(backbone: torch.nn.Module):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Device: ", device)

    classifier = SiameseMLP(backbone)
    classifier = classifier.to(device)
    print(classifier)

    criterion = nn.BCELoss()
    optimiser = optim.Adam(classifier.mlp.parameters(), lr=1e-3, betas=(0.9, 0.999))
    return classifier, criterion, optimiser, device

def load_from_checkpoint(filename:str, model:nn.Module, optimizer:optim.Optimizer):
    checkpoint = torch.load(RESULTS_PATH + filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    starting_epoch = checkpoint['epoch']
    training_loss = checkpoint['loss_train']
    eval_loss = checkpoint['loss_eval']
    print(f"Resuming {model.__class__.__name__} training from epoch {str(starting_epoch)}")
    return starting_epoch, model, optimizer, training_loss, eval_loss

def save_checkpoint(epoch:int, model:nn.Module, optimizer:optim.Optimizer, training_loss:list, eval_loss:list):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_train': training_loss,
        'loss_eval': eval_loss
    }, RESULTS_PATH + f"{model.__class__.__name__}_checkpoint.tar"
    )

def train_siamese_one_epoch(model: SiameseNeuralNet, 
                            criterion: nn.Module, 
                            optimiser: optim.Optimizer, 
                            device: torch.device,
                            train_loader: torch.utils.data.DataLoader):
    model.train()
    start = time.time()
    num_batches = len(train_loader)
    total_loss = 0.0
    loss_list = []
    for i, (x1, x2, label) in enumerate(train_loader, 0):
        # forward pass
        x1 = x1.to(device)
        x2 = x2.to(device)
        label = label.to(device)

        optimiser.zero_grad()
        x1_features, x2_features = model(x1, x2)
        loss = criterion(x1_features, x2_features, label)

        # Backward and optimize
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0:
            print(f"Step [{i+1}/{len(train_loader)}] Loss: {loss.item()}")

        total_loss += loss.item()
        
        if (i+1) % (num_batches // 10) == 0:
            loss_list.append(loss.item())

    mean_loss = total_loss / num_batches
    end = time.time()
    elapsed = end - start

    return loss_list, mean_loss, elapsed

def eval_siamese_one_epoch(model: SiameseNeuralNet,
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
        loss_list = []
        for i, (x1, x2, labels) in enumerate(test_loader, 0):
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            x1_features, x2_features = model(x1, x2)

            loss = criterion(x1_features, x2_features, labels)

            if (i+1) % 100 == 0:
                print(f"Step [{i+1}/{len(test_loader)}] Loss: {loss.item()}")

            total_loss += loss.item()

            if (i+1) % (num_batches // 10) == 0:
                loss_list.append(loss.item())
        
        mean_loss = total_loss / num_batches
        
    end = time.time()
    elapsed = end - start

    return loss_list, mean_loss, elapsed

def train_classifier_one_epoch(model: nn.Module,
                                criterion: nn.Module,
                                optimiser: optim.Optimizer,
                                device: torch.device,
                                train_loader: torch.utils.data.DataLoader):
    model.train()
    start = time.time()
    num_batches = len(train_loader)
    total_loss = 0.0
    loss_list = []
    for i, (x, label) in enumerate(train_loader, 0):
        # forward pass
        x = x.to(device)
        label = label.to(device)
        label = label.float()

        optimiser.zero_grad()
        out = model(x)
        out = out.view(-1)

        loss = criterion(out, label)

        # Backward and optimize
        loss.backward()
        optimiser.step()

        if (i+1) % 100 == 0:
            print(f"Step [{i+1}/{len(train_loader)}] Loss: {loss.item()}")

        total_loss += loss.item()

        if (i+1) % (num_batches // 10) == 0:
            loss_list.append(loss.item())

    mean_loss = total_loss / num_batches
    end = time.time()
    elapsed = end - start

    return loss_list, mean_loss, elapsed

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
        loss_list = []
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            labels = labels.float()

            outputs = model(images)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = torch.round(outputs)
            # _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i+1) % (num_batches // 10) == 0:
                loss_list.append(loss.item())

        print('Test Accuracy: {} %'.format(100 * correct / total))
        mean_loss = total_loss / num_batches
    
    end = time.time()
    elapsed = end - start

    return loss_list, mean_loss, elapsed

def test_loss():
    cont_loss = ContrastiveLoss()
    input1 = torch.rand(2, 4096)
    input2 = torch.rand(2, 4096)
    
    labels = torch.Tensor([1,0])
    old_loss = contrastive_loss(input1, input2, labels)
    new_loss = cont_loss(input1, input2, labels)
    print(old_loss)
    print(new_loss)

def Siamese_training(total_epochs:int, random_seed=None, checkpoint=None):
    Siamese_checkpoint_filename = checkpoint

    train_loader = load_data(training=True, Siamese=True, random_seed=random_seed)
    test_loader = load_data(training=False, Siamese=True, random_seed=random_seed)
    siamese_net, criterion, optimiser, device = initialise_Siamese_training()

    if Siamese_checkpoint_filename is not None:
        starting_epoch, siamese_net, optimiser, training_losses, eval_losses = load_from_checkpoint(Siamese_checkpoint_filename, siamese_net, optimiser)
    else:
        starting_epoch = 0
        training_losses, eval_losses = [], []

    print('starting training and validation loop for Siamese backbone')
    start = time.time()

    previous_best_loss = float('inf')
    for epoch in range(starting_epoch, total_epochs):
        print(f'Training Epoch {epoch+1}')
        loss_list, avg_train_loss, elapsed = train_siamese_one_epoch(siamese_net, criterion, optimiser, device, train_loader)
        training_losses += loss_list
        print(f'Training Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_train_loss:.4f}')

        print(f'Validating Epoch {epoch+1}')
        loss_list, avg_eval_loss, elapsed = eval_siamese_one_epoch(siamese_net, criterion, device, test_loader)
        eval_losses += loss_list
        print(f'Validating Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_eval_loss:.4f}')

        if avg_eval_loss < previous_best_loss:
            save_checkpoint(epoch + 1, siamese_net, optimiser, training_losses, eval_losses)
            previous_best_loss = avg_eval_loss

    end = time.time()
    elapsed = end - start
    print("Siamese Training and Validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    plt.figure(figsize=(10,5))
    plt.title("Training and Evaluation Loss During Training")
    plt.plot(training_losses, label="Train")
    plt.plot(eval_losses, label="Eval")
    plt.xlabel("Epochs / 10")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(RESULTS_PATH + f"Siamese_train_and_eval_loss_after_{total_epochs}_epochs.png")

    return siamese_net

def classifier_training(backbone: SiameseTwin, total_epochs:int, random_seed=None, checkpoint=None):
    classifier_checkpoint_filename = checkpoint

    train_loader = load_data(training=True, Siamese=False, random_seed=random_seed)
    test_loader = load_data(training=False, Siamese=False, random_seed=random_seed)

    classifier, criterion, optimiser, device = initialise_classifier_training(backbone)

    if classifier_checkpoint_filename is not None:
        starting_epoch, classifier, optimiser, training_losses, eval_losses = load_from_checkpoint(classifier_checkpoint_filename, classifier, optimiser)
    else:
        starting_epoch = 0
        training_losses, eval_losses = [], []

    print('starting training and validation loop for Classifier')
    start = time.time()

    previous_best_loss = float('inf')
    for epoch in range(starting_epoch, total_epochs):
        print(f'Training Epoch {epoch+1}')
        loss_list, avg_train_loss, elapsed = train_classifier_one_epoch(classifier, criterion, optimiser, device, train_loader)
        training_losses += loss_list
        print(f'Training Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_train_loss:.4f}')

        print(f'Validating Epoch {epoch+1}')
        loss_list, avg_eval_loss, elapsed = eval_classifier_one_epoch(classifier, criterion, device, test_loader)
        eval_losses += loss_list
        print(f'Validating Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_eval_loss:.4f}')

        if avg_eval_loss < previous_best_loss:
            save_checkpoint(epoch + 1, classifier, optimiser, training_losses, eval_losses)
            previous_best_loss = avg_eval_loss

    end = time.time()
    elapsed = end - start
    print("Classifier Training and Validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    plt.figure(figsize=(10,5))
    plt.title("Training and Evaluation Loss During Training")
    plt.plot(training_losses, label="Train")
    plt.plot(eval_losses, label="Eval")
    plt.xlabel("Epochs / 10")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(RESULTS_PATH + f"Classifier_train_and_eval_loss_after_{total_epochs}_epochs.png")

if __name__ == "__main__":
    # normal training workflow
    # net = Siamese_training(20, 69)
    # classifier_training(net.backbone, 20, 69)

    # training Siamese workflow
    Siamese_training(50, 69)

    # train classifier from existing Siamese model workflow
    checkpoint = "SiameseNeuralNet_checkpoint.tar"
    siamese_net, criterion, optimiser, device = initialise_Siamese_training()
    start_epoch, siamese_net, optimiser, training_losses, eval_losses = load_from_checkpoint(checkpoint, siamese_net, optimiser)
    print(f"best epoch: {start_epoch - 1}")
    print(f"training losses: {training_losses}")
    print(f"eval losses: {eval_losses}")

    classifier_training(siamese_net.backbone, 10, 69)

