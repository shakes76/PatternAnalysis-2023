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

# create data loader
train_loader, validation_loader, test_loader = dataset.load_data(
    train_folder_path, train_ad_path, train_nc_path, test_ad_path, test_nc_path, batch_size=32)

# define models
embbeding = modules.Embedding()
model = modules.SiameseNet(embbeding)
model.to(device)

# define loss function
margin = 1
criterion = modules.TripletLoss(margin)
optimizer = optim.Adam(model.parameters(), lr=0.001)


# train
def train(train_loader, epoches):

    for epoch in range(epoches):
        # set model to train mode
        model.train()

        total_accuracy = 0
        total_samples = 0

        for anchor, positive, negative in train_loader:
            # move data to gpu
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # front propagation
            emb_anchor, emb_positive, emb_negative = model(anchor, positive, negative)
            mask = modules.semi_hard_triplet_mining(emb_anchor, emb_positive, emb_negative, margin)

            if len(emb_anchor[mask]) == 0:
                del emb_anchor, emb_positive, emb_negative
                torch.cuda.empty_cache()
                continue

            loss = criterion(emb_anchor[mask], emb_positive[mask], emb_negative[mask])

            # calculate accuracy
            batch_accuracy = calculate_accuracy(emb_anchor, emb_positive, emb_negative)
            total_accuracy += batch_accuracy * anchor.size(0)
            total_samples += anchor.size(0)

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save the model after each epoch
        torch.save(model.state_dict(),
                    "D:/Study/GitHubDTClone/COMP3710A3/PatternAnalysis-2023/recognition/s4627382_SiameseNetwork/SiameseNet.pth")
        
        # calculate accuracy
        avg_accuracy = total_accuracy / total_samples
        
        print(f"Epoch [{epoch+1}/{epoches}], Loss: {loss.item():.4f}, Accuracy: {avg_accuracy:.4f}")

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
    ave_loss = total_loss/len(validation_loader)
    ave_accuracy = total_accuracy/len(validation_loader)
    print(f"Average loss: {ave_loss:.4f} \n Average accuracy: {ave_accuracy:.4f}")

    return ave_loss, ave_accuracy


# test
def test(test_loader):
    # set model to evaluation mode
    model.eval()
    # TODO


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
    # train the model
    epoches = 10
    print("Training")
    train(train_loader, epoches)

    # validate the model
    print("Validating")
    validate(validation_loader)
    

if __name__ == "__main__":
    main()