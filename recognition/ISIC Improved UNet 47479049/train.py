import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from modules import UNet
from dataset import get_loaders

# Hyperparameters
learning_rate = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
num_epochs = 16
num_workers = 4
image_height = 96
image_width = 128
test_dir = "data\ISIC2018_Task1-2_Test_Input"
test_out_dir = "data\ISIC2018_Task1-2_Test_Output"
train_dir = "data\ISIC2018_Task1-2_Training_Input_x2"
mask_dir = "data\ISIC2018_Task1_Training_GroundTruth_x2"

def train(loader, val_loader, model, optimizer, criterion, num_epochs):
    print("Training Start")
    validation_loss_history = []
    loss_history = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data, mask in loader:
            data = data.to(device)
            mask = mask.to(device)

            # Forward
            predictions = model(data)
            loss = criterion(predictions, mask)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        total_validation_loss = 0
        for data, mask in val_loader:
            data = data.to(device)
            mask = mask.to(device)

            predictions = model(data)
            validation_loss = criterion(predictions, mask)

            total_validation_loss += validation_loss.item()

        total_validation_loss = total_validation_loss/len(val_loader)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f}, Total Validation Loss: {total_validation_loss:.4f}, Dice Score of Validation: {(1-total_validation_loss):.4f}")
        validation_loss_history.append(total_validation_loss)
        loss_history.append(avg_loss)

    return validation_loss_history, loss_history


class diceloss(torch.nn.Module):
    """
    Calculates diceloss
    taken from: https://discuss.pytorch.org/t/implementation-of-dice-loss/53552
    """

    def init(self):
        super(diceloss, self).__init__()
    def forward(self, pred, target):
        smooth = 1
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat*iflat).sum()
        B_sum = torch.sum(tflat*tflat).sum()
        return 1 - ((2. * intersection + smooth)/ (A_sum + B_sum + smooth))
    

def loss_graph(results, results_val, num_epochs):
    plt.plot(range(1, num_epochs+1), results, 'k', label="Training")
    plt.plot(range(1, num_epochs+1), results_val, 'r', label="Validation")
    plt.title('Loss Graph')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("Lossgraph.jpg")
    plt.legend()
    plt.show()


def create_mask(loader, model, max_pics, device=device):
    """
    create_mask will create a mask given the laoder using the model given
    """

    model.eval()

    for idx, (x,y) in enumerate(loader):
        if idx > max_pics:
            break
        x = x.to(device=device)
        with torch.no_grad():
            pred = model(x)
            pred = (pred>0.5).float()
        torchvision.utils.save_image(
            pred, f"{y}"
        )
        

    model.train()



def main():
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_height, image_width), antialias=None)
    ])

    model = UNet(3, 1).to(device)
    criterion = diceloss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    train_loader, val_loader = get_loaders(
        train_dir,
        mask_dir,
        batch_size,
        train_transforms,
    )


    validation_loss_history, loss_history = train(train_loader, val_loader, model, optimizer, criterion, num_epochs)
    torch.save(model.state_dict(), "model")

    loss_graph(validation_loss_history, loss_history, num_epochs)


if __name__ == "__main__":
    main()



