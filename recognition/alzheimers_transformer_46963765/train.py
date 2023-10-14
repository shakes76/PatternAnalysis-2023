import dataset
from modules import *
import torch.optim as optim
import matplotlib.pyplot as plt
from predict import test_accuracy, visualize_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#put in training loop
def train(epochs, depth, learning_rate, batch_size):
    
    model = ADNI_Transformer(depth=depth)
    model.to(device=device)

    dataset = ds.ADNI_Dataset()
    train_loader = dataset.get_train_loader()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    criterion = nn.BCELoss()
    batch_losses = []

    # training loop
    for epoch in range(epochs):
        batch_loss = 0
        model.train()
        for j, (images, labels) in  enumerate(train_loader):
            if images.size(0) == batch_size:
                optimizer.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze().to(torch.float), labels.to(torch.float))
                loss.backward()
                optimizer.step()
                
                batch_loss += loss.item()
                print(batch_loss/(j+1))
                
        batch_losses.append(batch_loss/(j+1))
        print("epoch {} complete".format(epoch + 1))
        #print("loss is {}".format(batch_loss))
        #accuracy = test_accuracy(model)
        #print("accuracy is {}".format(accuracy))

        
    return model, batch_losses



if __name__ == "__main__":
    epochs = 10
    depth = 3
    learning_rate = 0.0005
    batch_size = 32
    
    model, losses = train(epochs, depth, learning_rate, batch_size)
    torch.save(model.state_dict(), "model/model.pth")
    
    accuracy = test_accuracy(model)
    print("Accuracy of model is {}%".format(accuracy))
    visualize_loss(losses)


    
