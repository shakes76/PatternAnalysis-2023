# containing the source code for training, validating, testing and saving your model. 
# The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”. 
# Make sure to plot the losses and metrics during training

import dataset, modules
import torch
import torch.optim as optim
from annoy import AnnoyIndex
device = torch.device('cuda')

# data path
train_folder_path = "D:/Study/MLDataSet/AD_NC/train"
train_ad_path = "D:/Study/MLDataSet/AD_NC/train/AD"
train_nc_path = "D:/Study/MLDataSet/AD_NC/train/NC"
test_ad_path = "D:/Study/MLDataSet/AD_NC/test/AD"
test_nc_path = "D:/Study/MLDataSet/AD_NC/test/NC"

# hyperparameters
margin = 0.5
epoches = 8

# create data loader
train_loader, validation_loader, test_loader = dataset.load_data(
    train_folder_path, train_ad_path, train_nc_path, test_ad_path, test_nc_path, batch_size=32)

# define models
embbeding = modules.Embedding()
model = modules.SiameseNet(embbeding)
model.to(device)

# define loss function
criterion = modules.TripletLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train
def train(train_loader, epoches):
    for epoch in range(epoches):
        # set model to train mode
        model.train()
        skip = 0
        total_accuracy = 0
        total_samples = 0
        highest_val_accuracy = 0

        for anchor, positive, negative in train_loader:
            # move data to gpu
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # front propagation
            emb_anchor, emb_positive, emb_negative = model(anchor, positive, negative)

            # mask = modules.semi_hard_triplet_mining(emb_anchor, emb_positive, emb_negative, int(len(emb_anchor) * 0.5), margin)
            mask = modules.semi_hard_triplet_mining(emb_anchor, emb_positive, emb_negative, 0, margin)
            if len(emb_anchor[mask]) == 0:
                skip += 1
                del emb_anchor, emb_positive, emb_negative
                torch.cuda.empty_cache()
                continue
        
            loss = criterion(emb_anchor[mask], emb_positive[mask], emb_negative[mask])
            # loss = criterion(emb_anchor, emb_positive, emb_negative)
            
            # calculate accuracy
            batch_accuracy = calculate_accuracy(emb_anchor, emb_positive, emb_negative)
            total_accuracy += batch_accuracy * anchor.size(0)
            total_samples += anchor.size(0)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # calculate accuracy
        avg_accuracy = total_accuracy / total_samples

        validate_loss, validate_accuracy = validate(validation_loader)

        print(f"Epoch [{epoch+1}/{epoches}], Loss: {loss.item():.4f}, Accuracy: {avg_accuracy:.4f}, validate loss: {validate_loss:.4f}, validate accuracy: {validate_accuracy:.4f}")
        
        # save the model
        if validate_accuracy > highest_val_accuracy:
            torch.save(model.state_dict(),
                       "D:/Study/GitHubDTClone/COMP3710A3/PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/SiameseNet.pth")
            highest_val_accuracy = validate_accuracy
            print("Model saved")

# validate
def validate(validation_loader):
    # set model to evaluation mode
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for anchor, positive, negative in validation_loader:
            # move data to gpu
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # front propagation
            emb_anchor, emb_positive, emb_negative = model(anchor, positive, negative)
            loss = criterion(emb_anchor, emb_positive, emb_negative)

            # get loss and accuracy
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(emb_anchor, emb_positive, emb_negative)
    
    # calculate average loss and average accuracy
    validate_loss = total_loss/len(validation_loader)
    validate_accuracy = total_accuracy/len(validation_loader)

    return validate_loss, validate_accuracy


# test
def test(test_loader):
    # set model to evaluation mode
    model.eval()
    total_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for anchor, positive, negative in test_loader:
            # move data to gpu
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # front propagation
            emb_anchor, emb_positive, emb_negative = model(anchor, positive, negative)
            loss = criterion(emb_anchor, emb_positive, emb_negative)

            # get loss and accuracy
            total_loss += loss.item()
            total_accuracy += calculate_accuracy(emb_anchor, emb_positive, emb_negative)
    
    # calculate average loss and average accuracy
    test_loss = total_loss/len(test_loader)
    test_accuracy = total_accuracy/len(test_loader)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy



# calculate accuracy, 
# if anchor_positive distance < threshold, these two samples will be considered similar
# if anchor_negative distance > threshold, these two samples will be considered unsimilar
def calculate_accuracy(anchor, positive, negative, threshold=0.5):
    
    # calculate the distance from anchor to positive and negative samples
    # .sum(1) will reduce a dimension, so the shape of anchor_positive: [batch_size, 2] -> [batch_size]
    anchor_positive = (anchor - positive).pow(2).sum(1).sqrt()
    anchor_negative = (anchor - negative).pow(2).sum(1).sqrt()

    # correct predictions
    true_positive = (anchor_positive < threshold).float()
    true_negative = (anchor_negative > threshold).float()

    # total are true_positive + true_negative + false_positive +  false_negative, 
    # they summing up equal to true_positive.size(0) + true_negative.size(0)
    total = true_positive.size(0) + true_negative.size(0)
    accuracy = (true_positive.sum().item() + true_negative.sum().item())/total
    
    return accuracy

def main():
    train_mode = 1

    if train_mode:
        # train the model
        print("Training")
        train(train_loader, epoches)

    else:
        print("Testing")
        model.load_state_dict(
        torch.load("D:/Study/GitHubDTClone/COMP3710A3/PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/SiameseNet.pth"))
        test(test_loader)

    print("Done")
    

if __name__ == "__main__":
    main()