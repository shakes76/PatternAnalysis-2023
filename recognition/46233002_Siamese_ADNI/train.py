import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_metric_learning import losses, miners
import time

def train_encoder(model, train_loader, device, num_epochs, tensor_path, learning_rate):

    # Initialise loss functions, optimizer, and scheduler
    miner = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.EASY, 
                                      neg_strategy=miners.BatchEasyHardMiner.SEMIHARD)
    criterion = losses.TripletMarginLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=learning_rate,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=num_epochs)

    print("Begin Training Encoder")
    model.train()
    start = time.time()

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            total_loss = 0.0
            for i, (images, labels) in enumerate(train_loader):
                # Retreive data and move to device
                images = images.float().to(device)
                labels = labels.long().to(device)

                # Forward pass
                outputs = model(images)
                hard_pairs = miner(outputs, labels)

                # Save final embeddings
                if (epoch == num_epochs - 1):
                    torch.save({"embeddings": outputs, "labels": labels}, 
                            tensor_path+f"/embedding{i}.pt")

                # Compute loss
                loss = criterion(outputs, labels, hard_pairs)
                #loss = criterion(outputs, labels)
                total_loss += loss

                optimizer.zero_grad() # Zero gradient
                loss.backward()       # Back propagate loss
                optimizer.step()      # Optimizer takes a step
                scheduler.step()      # Scheduler takes a step 

                # Print process
                if i % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(
                        epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            
            # Print total loss of an epoch
            print("Epoch [{}], Total Loss: {:.5f}".format(epoch+1, total_loss))
 
    end = time.time()
    elapsed = end - start
    print("End Training")
    print(f"Training took {str(elapsed)} secs or {str(elapsed/60)} mins")
    print()

def train_classifier(model, train_loader, device, num_epochs, 
                     learning_rate):
    
    # Initialise loss functions, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=learning_rate,
                                                    steps_per_epoch=len(train_loader),
                                                    epochs=num_epochs)

    print("Begin Training Classifier")
    model.train()
    start = time.time()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            total_loss = 0.0
            for i, (embeddings, labels) in enumerate(train_loader):
                # Retreive data and move to device
                embeddings = embeddings.squeeze().float().to(device)
                labels = labels.squeeze().long().to(device)

                # Forward pass
                outputs = model(embeddings)

                # Back propagation
                loss = criterion(outputs, labels)
                total_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Print process
                if i % 100 == 0:
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(
                        epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            print("Epoch [{}], Total Loss: {:.5f}".format(epoch+1, total_loss))

    end = time.time()
    elapsed = end - start
    print("End Training")
    print(f"Training took {str(elapsed)} secs or {str(elapsed/60)} mins")
    print()
    print(outputs)
    print(labels)
    preds = torch.argmax(outputs, axis = 1)
    print(preds)


def validate_model(encoder, classifier, val_loader, device):

    # Initialise loss functions
    criterion_encoder = losses.TripletMarginLoss()
    criterion_classifier = nn.CrossEntropyLoss()

    # Initialise variables for counting
    total_loss_encoder = 0.0
    total_loss_classifier = 0.0
    num_correct = 0
    total = 0

    print("Begin Validation")
    encoder.eval()
    classifier.eval()
    start = time.time()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            # Retreive data and move to device
            images = images.float().to(device)
            labels = labels.long().to(device)

            # Forward data
            embeddings = encoder(images)
            outputs = classifier(embeddings)

            # Compute losses 
            loss_encoder = criterion_encoder(outputs, labels)
            loss_classifier = criterion_classifier(outputs, labels)
            total_loss_encoder += loss_encoder
            total_loss_classifier += loss_classifier

            preds = torch.argmax(outputs, axis = 1)
            num_correct += (preds == labels).sum()
            total += torch.numel(preds)
            accuracy = num_correct / total * 100

            if i % 10 == 0:
                print("Step [{}/{}] Encoder Loss: {:.5f} Classifier Loss: {:.5f}".format(
                    i+1, len(val_loader), loss_encoder.item(), loss_classifier.item()))
                print("Got {}/{} with acc {:.2f}".format(num_correct, total, accuracy))

    end = time.time()
    elapsed = end - start
    print("End Validation")
    print("Validation took {} secs or {} mins".format(str(elapsed), str(elapsed/60)))
    print("Total Encoder Loss: {:.5f} Total Classifier Loss: {:.5f}".format(total_loss_encoder, total_loss_classifier))
    print("Got {}/{} with acc {:.2f}".format(num_correct, total, accuracy))
    print()


def test_model(encoder, classifier, test_loader, device):
    print("Begin Testing")
    encoder.eval()
    classifier.eval()

    start = time.time()
    num_correct = 0
    total = 0
    with torch.no_grad():
        for (images, labels) in test_loader:
            # Retreive data and move to device
            images = images.float().to(device)
            labels = labels.long().to(device)

            # Forward data
            embeddings = encoder(images)
            outputs = classifier(embeddings)
            preds = torch.argmax(outputs, axis = 1)
            num_correct += (preds == labels).sum()
            total += torch.numel(preds)
            accuracy = num_correct / total * 100

    end = time.time()
    elapsed = end - start
    print("End Testing")
    print(f"Testing took {str(elapsed)} secs or {str(elapsed/60)} mins")
    print(f"Got {num_correct}/{total} with acc {accuracy:.2f}")
    print()
    print(labels)
    print(preds)
