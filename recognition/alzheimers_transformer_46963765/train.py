import dataset as ds
from modules import *
import torch.optim as optim
from predict import test_accuracy, visualize_loss, visualize_accuracies

# check for CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epochs, learning_rate, batch_size, model):
    # retrieve dataset
    dataset = ds.ADNI_Dataset(); train_loader = dataset.get_train_loader()
    
    # BCELoss used with the Adam optimiser
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.BCELoss()
    
    # lists to store the results of training
    batch_losses = []
    accuracies = []

    # main training loop
    for epoch in range(epochs):
        batch_loss = 0
        # iterate through the loeader in train mode
        model.train()
        for j, (images, labels) in  enumerate(train_loader):
            # model cannot take batches smaller than specified batch size
            if images.size(0) == batch_size:
                # forward pass through model
                optimizer.zero_grad()
                images = images.to(device); labels = labels.to(device)
                outputs = model(images)
                # calculate loss and backpropogate
                loss = criterion(outputs.to(torch.float), labels.to(torch.float))
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                
        # Print results for the epoch        
        batch_losses.append(batch_loss/(j+1))
        print("epoch {} complete".format(epoch + 1))
        print("loss is {}".format(batch_loss/(j+1)))
        # Get accuracy for the epoch
        accuracy = test_accuracy(model, batch_size)
        print("accuracy is {}".format(accuracy))
        accuracies.append(accuracy) 
        
    return model, batch_losses, accuracies


if __name__ == "__main__":
    # hyper parameters
    epochs = 10
    depth = 3
    learning_rate = 0.0005
    batch_size = 32
    # model parameters
    LATENT_DIM = 32
    LATENT_EMB = 64
    latent_layers = 4
    latent_heads = 8
    classifier_out = 16
    batch_size = 32
    
    # instantiate the model and move it to GPU
    model = ADNI_Transformer(depth, LATENT_DIM, LATENT_EMB, latent_layers, latent_heads, classifier_out, batch_size)
    model.to(device=device)
    
    # train and save the model
    model, losses, train_accuracies = train(epochs, learning_rate, batch_size, model)
    torch.save(model.state_dict(), "model/model.pth")
    
    # visualise the results 
    print("testing final model accuracy")
    accuracy = test_accuracy(model, batch_size)
    print("Accuracy of model is {}%".format(accuracy))
    visualize_loss(losses)
    visualize_accuracies(train_accuracies)


    
