import torch
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt

def predict(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            y_hat = model(data)
            _, preds = torch.max(y_hat, 1)
            
            # plot images in a grid with labels above each image
            fig, axs = plt.subplots(2, 5, figsize=(10, 5))
            fig.suptitle('Batch Predictions')
            for i in range(2):
                for j in range(5):
                    idx = i * 5 + j
                    axs[i, j].imshow(data[idx].permute(1, 2, 0).cpu(), cmap="gray")
                    axs[i, j].set_title(f'Label: {target[idx].item()}, Pred: {preds[idx].item()}')
                    axs[i, j].axis('off')
            plt.show()


def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CrossEntropyLoss()

    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.type(torch.FloatTensor).to(device), y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")
    