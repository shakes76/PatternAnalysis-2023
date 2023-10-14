import torch
import torch.nn as nn
from torch import optim
from torch.hub import tqdm
from dataset import DataLoader
from modules import ViT

class HyperParameters(object):
    '''
    HyperParameters
    
    This class defines all the hyperparameters for 
    the vision transformer
    '''

    def __init__(self) -> None:
        self.patch_size = 16
        self.latent_size = 768
        self.n_channels = 3
        self.num_encoders = 16
        self.num_heads = 12
        self.dropout = 0.1
        self.num_classes = 2
        self.epochs = 10
        self.lr = 2e-4
        self.weight_decay = 3e-2
        self.batch_size = 64
        self.dry_run = False

def check_accuracy(loader, model, device):
    '''
    Check the accuracy of the model on the given dataloader
    '''
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    model.train()
    return num_correct / num_samples


def main():

    args = HyperParameters()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    dl = DataLoader(batch_size=args.batch_size)
    train_loader = dl.get_training_loader()
    test_loader = dl.get_test_loader()
    total_loss = 0.0

    model = ViT(args).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(args.epochs):
        tk = tqdm(train_loader, desc="EPOCH" + "[TRAIN]" 
                  + str(epoch + 1) + "/" + str(args.epochs))
        
        for batch_idx, (data, targets) in enumerate(tk):
            # Get data to cuda
            data = data.to(device=device)
            targets = targets.to(device=device)

            # Forward propogation
            scores = model(data)
            loss = criterion(scores, targets)

            # Back propogation
            optimizer.zero_grad()
            loss.backward()

            # Optimizer step
            optimizer.step()

            total_loss += loss.item()
            tk.set_postfix({"Loss": "%6f" % float(total_loss / (batch_idx + 1))})
    
    print(f"Accuracy on training set: {check_accuracy(train_loader, model, device)*100:.2f}")
    print(f"Accuracy on test set: {check_accuracy(test_loader, model, device)*100:.2f}")

if __name__ == "__main__":
    main()