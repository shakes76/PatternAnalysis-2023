import dataset
from modules import *
import torch.optim as optim
import matplotlib.pyplot as plt



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#put in training loop
def train(epochs, depth):
    model = ADNI_Transformer(depth=depth)
    model.to(device=device)

    dataset = ds.ADNI_Dataset()
    train_loader = dataset.get_train_loader()
    optimizer = optim.Adam(model.parameters(), 1e-5)
    criterion = nn.BCELoss()

    batch_losses = []

    for epoch in range(epochs):
        batch_loss = 0

        model.train()
        for j, (images, labels) in  enumerate(train_loader):
            if images.size(0) == 32:
                optimizer.zero_grad()
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                
                loss = criterion(outputs.squeeze().to(torch.float), labels.to(torch.float))
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                
        batch_losses.append(batch_loss)
        print("epoch {} complete".format(epoch + 1))
        print("loss is {}".format(batch_loss))
        
    return model, batch_losses


def visualize_loss(batch_losses):
    
    epochs = range(1, len(batch_losses) + 1)

    
    plt.plot(epochs, batch_losses, marker='o', linestyle='-')
    plt.xlabel('Epoch')
    plt.ylabel('Batch Loss')
    plt.title('Batch Loss Over Epochs')
    plt.grid(True)
    plt.savefig('plots/loss_plot.png')

    plt.show()


def test_accuracy(model):
    dataset = ds.ADNI_Dataset()
    test_laoder = dataset.get_test_loader()
        
    correct_predictions = 0
    total_samples = 0    
        
    model.eval() 
    for j, (images, labels) in  enumerate(test_laoder):
        if images.size(0) == 32:

            images = images.to(device) 
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = (outputs >= 0.5).squeeze().long()
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy

if __name__ == "__main__":
    epochs = 15
    depth = 6
    # 15 epochs and a depth of 6
    model, losses = train(epochs, depth)
    torch.save(model.state_dict(), "model/model.pth")
    
    accuracy = test_accuracy(model)
    print("Accuracy of model is {}%".format(accuracy))
    
    visualize_loss(losses)
    
# check depth (is 6)
# check skip connections
# check latent_transformer layers and heads (are 8)

    
