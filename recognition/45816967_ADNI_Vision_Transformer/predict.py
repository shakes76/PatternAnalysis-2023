
import argparse
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from dataset import generate_adni_datasets

def predict(model, dataloader, dim=(2, 5), version_prefix="vit", save_fig=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define labels
    labels = ["CN", "AD"]

    model.eval()
    with torch.no_grad():
        # get random data subset
        (data, target) = next(iter(dataloader))
        data, target = data.to(device), target.to(device)
        y_hat = model(data)
        _, preds = torch.max(y_hat, 1)
        
        # plot images in a grid with labels above each image
        fig, axs = plt.subplots(dim[0], dim[1], figsize=(2*dim[1], 2*dim[0]))
        fig.suptitle(f"{version_prefix} Predictions")
        for i in range(dim[0]):
            for j in range(dim[1]):
                # Plot each image in grid with true and predicted labels
                idx = i * dim[1] + j
                axs[i, j].imshow(data[idx].permute(1, 2, 0).cpu(), cmap="gray")
                axs[i, j].set_title(f'Label: {labels[target[idx].item()]}, Pred: {labels[preds[idx].item()]}')
                axs[i, j].axis('off')
                axs[i, j].title.set_size(8)
        #save image
        if (save_fig):
            plt.savefig(f"{version_prefix}_preds.png")
        plt.show()
    
    
def main():
    # Parse commandline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", nargs='?', default=None, type=str)
    parser.add_argument("--dim", nargs='?', default=4, type=int)
    args = parser.parse_args()
    
    # Check if model directory is specified
    if args.model_dir is None:
        raise("Must specify model directory")
    else:
        print("model_dir: ", args.model_dir, "dim: ", args.dim)
        # Load Data and model
        train_set, val_set, test_set = generate_adni_datasets(datasplit=0.1)
        model = torch.load(args.model_dir)

        # Create dataloader
        test_loader = DataLoader(test_set, shuffle=True, batch_size=args.dim**2 + 1)
        
        # Run predictions
        predict(model, dataloader=test_loader, dim=(args.dim, args.dim), version_prefix="vit")

if __name__ == "__main__":
	main()