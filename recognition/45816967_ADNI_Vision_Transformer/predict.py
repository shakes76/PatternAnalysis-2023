import torch
import matplotlib.pyplot as plt

def predict(model, dataloader, device):
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
