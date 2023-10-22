import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_metric_learning import losses, miners
import time

def train_encoder(encoder, train_loader, val_loader, device, num_epochs, tensor_path, learning_rate):

    # Initialise loss functions, optimizer, and scheduler
    miner = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.EASY, 
                                      neg_strategy=miners.BatchEasyHardMiner.SEMIHARD)
    criterion = losses.TripletMarginLoss(margin=0.1)
    optimizer = optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=num_epochs)

    print("Begin Training Encoder")
    start = time.time()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            # Train 
            encoder.train()
            total_train_loss = 0.0
            if epoch > 16:
                miner = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.HARD, 
                                                  neg_strategy=miners.BatchEasyHardMiner.HARD)
            elif epoch > 12:
                miner = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.SEMIHARD, 
                                                  neg_strategy=miners.BatchEasyHardMiner.HARD)
            elif epoch > 8:
                miner = miners.BatchEasyHardMiner(pos_strategy=miners.BatchEasyHardMiner.EASY, 
                                                  neg_strategy=miners.BatchEasyHardMiner.HARD)
            for i, (images, labels) in enumerate(train_loader):
                # Retreive data and move to device
                images = images.float().to(device)
                labels = labels.long().to(device)
                # Forward pass & generaet triplets 
                outputs = encoder(images)
                hard_pairs = miner(outputs, labels)
                # Save final embeddings
                if (epoch == num_epochs - 1):
                    torch.save({"embeddings": outputs, "labels": labels}, 
                            tensor_path+f"/embedding{i}.pt")
                # Back propagation
                optimizer.zero_grad() # Zero gradient
                loss = criterion(outputs, labels, hard_pairs) # Compute loss
                total_train_loss += loss.item()
                loss.backward()       # Back propagate loss
                optimizer.step()      # Optimizer takes a step
                scheduler.step()      # Scheduler takes a step 
                # Print process
                #if i % 100 == 0:
                #    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(
                #        epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            
            # Validate
            encoder.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    # Retreive data and move to device
                    images = images.float().to(device)
                    labels = labels.long().to(device)
                    # Forward pass & generaet triplets 
                    outputs = encoder(images)
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()

            # Print summary
            print("Epoch [{}/{}], Train Loss: {:.5f}, Validation Loss: {:.5f}".format(
                epoch+1, num_epochs, total_train_loss/len(train_loader), total_val_loss/len(val_loader)))
    
    end = time.time()
    elapsed = end - start
    print("End Training")
    print(f"Training took {str(elapsed)} secs or {str(elapsed/60)} mins\n")

def train_classifier(encoder, classifier, train_loader, val_loader, device, num_epochs, learning_rate):
    
    # Initialise loss functions, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=num_epochs)

    print("Begin Training Classifier")
    start = time.time()
    encoder.eval()
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(num_epochs):
            # Train
            classifier.train()
            total_train_loss = 0.0
            for i, (embeddings, labels) in enumerate(train_loader):
                # Retreive data and move to device
                embeddings = embeddings.squeeze().float().to(device)
                labels = labels.squeeze().long().to(device)
                # Forward pass
                outputs = classifier(embeddings)
                # Back propagation
                optimizer.zero_grad() # Zero gradient
                loss = criterion(outputs, labels) # Compute loss
                total_train_loss += loss.item()
                loss.backward()       # Back propagate loss
                optimizer.step()      # Optimizer takes a step
                scheduler.step()      # Scheduler takes a step
                # Print process
                #if i % 100 == 0:
                #    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}".format(
                #        epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
            
            # Validate 
            classifier.eval()
            total_val_loss = 0.0
            num_correct = 0
            total = 0
            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    # Retreive data and move to device
                    images = images.float().to(device)
                    labels = labels.long().to(device)
                    # Forward data
                    embeddings = encoder(images)
                    outputs = classifier(embeddings)
                    # Compute losses
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    # Compute accuracy
                    preds = torch.argmax(outputs, axis = 1)
                    num_correct += (preds == labels).sum()
                    total += torch.numel(preds)
                    accuracy = num_correct / total * 100
            # Print summary
            print("Epoch [{}/{}], Train Loss: {:.5f}, Validation Loss: {:.5f}, Accuracy {:.2f}".format(
                epoch+1, num_epochs, total_train_loss/len(train_loader), total_val_loss/len(val_loader), accuracy))
    end = time.time()
    elapsed = end - start
    print("End Training")
    print(f"Training took {str(elapsed)} secs or {str(elapsed/60)} mins\n")

def validate_model(encoder, classifier, val_loader, device):
    num_correct = 0
    total = 0
    print("Begin Validation")
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            # Retreive data and move to device
            images = images.float().to(device)
            labels = labels.long().to(device)
            # Forward data
            embeddings = encoder(images)
            outputs = classifier(embeddings)
            # Get predictions
            preds = torch.argmax(outputs, axis = 1)
            num_correct += (preds == labels).sum()
            total += torch.numel(preds)
            accuracy = num_correct / total * 100
            if i % 10 == 0:
                print("Got {}/{} with acc {:.2f}".format(num_correct, total, accuracy))
    print("End Validation")
    print("Got {}/{} with acc {:.2f}\n".format(num_correct, total, accuracy))


def test_model(encoder, classifier, test_loader, device):
    num_correct = 0
    total = 0
    print("Begin Testing")
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for (images, labels) in test_loader:
            # Retreive data and move to device
            images = images.float().to(device)
            labels = labels.long().to(device)
            # Forward data
            embeddings = encoder(images)
            outputs = classifier(embeddings)
            # Get predictions
            preds = torch.argmax(outputs, axis=1)
            num_correct += (preds == labels).sum()
            total += torch.numel(preds)
            accuracy = num_correct / total * 100
    print("End Testing")
    print("Got {}/{} with acc {:.2f}\n".format(num_correct, total, accuracy))

def predict(encoder, classifier, pred_loader, device):
    num_ad = 0
    num_nc = 0
    total = 0
    predictions = []
    print("Begin Prediction")
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        for images in pred_loader:
            images = images.float().to(device)
            embeddings = encoder(images)
            outputs = classifier(embeddings)
            preds = torch.argmax(outputs, axis=1)
            num_ad += (preds == 1).sum()
            num_nc += (preds == 0).sum()
            total += torch.numel(preds)
            predictions.extend(preds.cpu().detach().numpy().tolist())
    print("End Prediction")
    print(f"Out of {total}\nNumber of AD = {num_ad}. Number of NC = {num_nc}")
    print("Predictions:")
    print(predictions)
