import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

import CONSTANTS
from modules import SiameseTwin, SiameseNeuralNet, SimpleMLP
from dataset import PairedDataset, load_data, load_test_data
from utils import load_from_checkpoint, save_checkpoint
from predict import make_predictions, visualise_sample_predictions


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

def test_loss():
    cont_loss = ContrastiveLoss()
    input1 = torch.rand(2, 4096)
    input2 = torch.rand(2, 4096)
    
    labels = torch.Tensor([1,0])
    old_loss = contrastive_loss(input1, input2, labels)
    new_loss = cont_loss(input1, input2, labels)
    print(old_loss)
    print(new_loss)

# def weights_initialisation(model:nn.Module):
#     classname = model.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.01)
#         nn.init.normal_(model.bias.data)
#     elif classname.find('Linear') != -1:
#         nn.init.normal_(model.weight.data, 0.0, 0.2)


def initialise_Siamese_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Device: ", device)

    siamese_net = SiameseNeuralNet()
    siamese_net = siamese_net.to(device)
    print(siamese_net)

    criterion = ContrastiveLoss(margin=1.0)
    optimiser = optim.Adam(siamese_net.parameters(), lr=0.0001, betas=(0.5, 0.999))
    return siamese_net, criterion, optimiser, device

def initialise_classifier_training():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Device: ", device)
    
    classifier = SimpleMLP()
    classifier = classifier.to(device)
    print(classifier)

    criterion = nn.BCELoss()
    optimiser = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.5, 0.999))
    return classifier, criterion, optimiser, device

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
                                backbone: SiameseTwin,
                                criterion: nn.Module,
                                optimiser: optim.Optimizer,
                                device: torch.device,
                                train_loader: torch.utils.data.DataLoader):
    model.train()
    backbone.eval()
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
        with torch.no_grad():
            out = backbone(x)
        out = model(out)
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
                                backbone: SiameseTwin,
                                criterion: nn.Module,
                                device: torch.device,
                                test_loader: torch.utils.data.DataLoader):
    model.eval()
    backbone.eval()
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

            outputs = backbone(images)
            outputs = model(outputs)
            outputs = outputs.view(-1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if (i+1) % (num_batches // 10) == 0:
                loss_list.append(loss.item())

        accuracy = correct / total

        print('Accuracy: {} %'.format(100 * accuracy))
        mean_loss = total_loss / num_batches
    
    end = time.time()
    elapsed = end - start

    return loss_list, mean_loss, elapsed, accuracy

def Siamese_training(total_epochs:int, random_seed=None, checkpoint=None):
    if random_seed is not None:
        torch.use_deterministic_algorithms(True)

    Siamese_checkpoint_filename = checkpoint

    train_loader = load_data(training=True, Siamese=True, random_seed=random_seed)
    validation_loader = load_data(training=False, Siamese=True, random_seed=random_seed)
    # validation_loader = load_test_data(Siamese=True, random_seed=random_seed)
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
        training_losses += [avg_train_loss]
        print(f'Training Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_train_loss:.4f}')

        print(f'Validating Epoch {epoch+1}')
        loss_list, avg_eval_loss, elapsed = eval_siamese_one_epoch(siamese_net, criterion, device, validation_loader)
        eval_losses += [avg_eval_loss]
        print(f'Validating Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_eval_loss:.4f}')

        if avg_eval_loss < previous_best_loss:
            # save_checkpoint(epoch + 1, siamese_net, optimiser, training_losses, eval_losses)
            previous_best_loss = avg_eval_loss
            print(f"loss improved in epoch {epoch + 1}")

    # save_checkpoint(epoch + 1, siamese_net, optimiser, training_losses, eval_losses)
    end = time.time()
    elapsed = end - start
    print("Siamese Training and Validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    plt.figure(figsize=(10,5))
    plt.title("Training and Evaluation Loss During Training")
    plt.plot(training_losses, label="Train")
    plt.plot(eval_losses, label="Eval")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(CONSTANTS.RESULTS_PATH + f"DSiamese_train_and_eval_loss_after_{total_epochs}_epochs.png")

    return siamese_net

def classifier_training(backbone: SiameseTwin, total_epochs:int, random_seed=None, checkpoint=None):
    if random_seed is not None:
        torch.use_deterministic_algorithms(True)

    classifier_checkpoint_filename = checkpoint

    train_loader = load_data(training=True, Siamese=False, random_seed=random_seed)
    validation_loader = load_data(training=False, Siamese=False, random_seed=random_seed)
    test_loader = load_test_data(Siamese=False, random_seed=random_seed)

    classifier, criterion, optimiser, device = initialise_classifier_training()

    if classifier_checkpoint_filename is not None:
        starting_epoch, classifier, optimiser, training_losses, eval_losses = load_from_checkpoint(classifier_checkpoint_filename, classifier, optimiser)
    else:
        starting_epoch = 0
        training_losses, eval_losses = [], []

    validation_accuracy = []
    test_accuracy = []
    print('starting training and validation loop for Classifier')
    start = time.time()

    best_loss = float('inf')
    best_accuracy = 0.5
    for epoch in range(starting_epoch, total_epochs):
        print(f'Training Epoch {epoch+1}')
        loss_list, avg_train_loss, elapsed = train_classifier_one_epoch(classifier, backbone, criterion, optimiser, device, train_loader)
        training_losses += [avg_train_loss]
        print(f'Training Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_train_loss:.4f}')

        print(f'Validating Epoch {epoch+1}')
        loss_list, avg_eval_loss, elapsed, accuracy = eval_classifier_one_epoch(classifier, backbone, criterion, device, validation_loader)
        eval_losses += [avg_eval_loss]
        validation_accuracy += [accuracy]
        print(f'Validating Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_eval_loss:.4f}')

        print(f'Testing Epoch {epoch+1}')
        loss_list, avg_eval_loss, elapsed, accuracy = eval_classifier_one_epoch(classifier, backbone, criterion, device, test_loader)
        test_accuracy += [accuracy]
        print(f'Testing Epoch {epoch+1} took {elapsed:.1f} seconds. Average loss: {avg_eval_loss:.4f}')

        if avg_eval_loss < best_loss:
            # save_checkpoint(epoch + 1, classifier, optimiser, training_losses, eval_losses)
            best_loss = avg_eval_loss
            print(f"loss improved in epoch {epoch + 1}")

        if accuracy > best_accuracy:
            # save_checkpoint(epoch + 1, classifier, optimiser, training_losses, eval_losses)
            best_accuracy = accuracy
            print(f"accuracy improved in epoch {epoch + 1}")

    # save_checkpoint(epoch + 1, classifier, optimiser, training_losses, eval_losses)
    end = time.time()
    elapsed = end - start
    print("Classifier Training and Validation took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss During Classifier Training")
    plt.plot(training_losses, label="Train")
    plt.plot(eval_losses, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(CONSTANTS.RESULTS_PATH + f"Classifier_loss_after_{total_epochs}_epochs.png")

    plt.figure(figsize=(10,5))
    plt.title("Testing and Validation Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.plot(test_accuracy, label="Testing Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(CONSTANTS.RESULTS_PATH + f"Classifier_Accuracy_after_{total_epochs}_epochs.png")
    return classifier

if __name__ == "__main__":
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # sequential training workflow
    net = Siamese_training(20, 42) # Model saving options available by uncommenting code in this function
    classifier = classifier_training(net.backbone, 20, 42) # comment out this line to train classifier based on last saved Siamese model

    # training classifier from existing Siamese model workflow
    checkpoint = "SiameseNeuralNet_checkpoint.tar"
    siamese_net, criterion, optimiser, device = initialise_Siamese_training()
    start_epoch, siamese_net, optimiser, training_losses, eval_losses = load_from_checkpoint(checkpoint, siamese_net, optimiser)

    classifier_training(siamese_net.backbone, 20, 35)

