import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
from modules import ViT
from dataset import generate_adni_datasets


def train(model, train_loader, val_loader, criterion=nn.CrossEntropyLoss(), n_epochs=50, lr=0.000025, version_prefix="vit0", gen_plots=True):
	
	# Defining model and training options
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")
	
	train_losses = []
	val_losses = []
	train_accs = []
	val_accs = []
	# Training loop
	optimizer = optim.Adam(model.parameters(), lr=lr)
	for epoch in range(n_epochs):
			model.train()
			correct, total = 0, 0
			train_loss = 0.0
			# Iterate for each batch
			for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
					# Format batch
					x, y = batch
					x, y = x.type(torch.FloatTensor).to(device), y.to(device)
					# Get prediction
					y_hat = model(x)

					# Calculate loss, accuracy, and backpropagate
					loss = criterion(y_hat, y)

					train_loss += loss.detach().cpu().item() / len(train_loader)

					correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
					total += len(x)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
			
			train_acc = correct / total * 100
			train_accs.append(train_acc)
			print(f"Epoch {epoch + 1}/{n_epochs} loss: {train_loss:.2f}")
			print(f"Train accuracy: {correct / total * 100:.2f}%")
			
			# Test loop
			model.eval()
			with torch.no_grad():
					correct, total = 0, 0
					val_loss = 0.0
					# Iterate for each batch
					for batch in tqdm(val_loader, desc="Validation"):
							# Format batch
							x, y = batch
							x, y = x.type(torch.FloatTensor).to(device), y.to(device)
       
							# Get prediction
							y_hat = model(x)
       
							# Calculate loss, accuracy
							loss = criterion(y_hat, y)
							val_loss += loss.detach().cpu().item() / len(val_loader)

							correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
							total += len(x)
					val_acc = correct / total * 100
					val_accs.append(val_acc)
					print(f"Val loss: {val_loss:.2f}")
					print(f"Val accuracy: {correct / total * 100:.2f}%")
			# Save model
			torch.save(model, f"models/{version_prefix}_model_{epoch + 1}_{val_acc}.pth")
			train_losses.append(train_loss)
			val_losses.append(val_loss)
	
	# Generate Train and Validation, Loss and Accuracy plots
	if gen_plots:
		plt.plot(train_losses, label="Train loss")
		# label the plot
		plt.title("Training loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		# save figure
		plt.savefig(f"{version_prefix}_trainloss.png")
		plt.show()


		plt.plot(val_losses, label="Validation loss")
		# label the plot
		plt.title("Validation loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")
		plt.legend()
		# save figure
		plt.savefig(f"{version_prefix}_valloss.png")
		plt.show()

		plt.plot(val_accs, label="Validation accuracy")
		# label the plot
		plt.title("Validation accuracy")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy (%)")
		plt.legend()
		# save figure
		plt.savefig(f"{version_prefix}_valacc.png")
		plt.show()
		
		plt.plot(train_accs, label="Train accuracy")
		# label the plot
		plt.title("Train accuracy")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy (%)")
		plt.legend()
		# save figure
		plt.savefig(f"{version_prefix}_trainacc.png")
		plt.show()
	return model, train_losses, val_losses, train_accs, val_accs


def test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = CrossEntropyLoss()
	
	# Evaluate model on test set
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        # Iterate for each batch
        for batch in tqdm(test_loader, desc="Testing"):
            # Format batch
            x, y = batch
            x, y = x.type(torch.FloatTensor).to(device), y.to(device)
            
            # Calculate prediction
            y_hat = model(x)
       
			# Calculate loss, accuracy
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")

def main():
    # Parse commandline arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--img_size", nargs='?', default=224, type=int)
	parser.add_argument("--patch_size", nargs='?', default=16, type=int)
	parser.add_argument("--num_layers", nargs='?', default=12, type=int)
	parser.add_argument("--embed_dim", nargs='?', default=256, type=int)
	parser.add_argument("--mlp_size", nargs='?', default=256, type=int)
	parser.add_argument("--num_heads", nargs='?', default=16, type=int)
	parser.add_argument("--n_epochs", nargs='?', default=50, type=int)
	parser.add_argument("--batch_size", nargs='?', default=64, type=int)
	args = parser.parse_args()

	print("img_size: ", args.img_size, "patch_size: ", args.patch_size, "num_layers: ", args.num_layers, "embed_dim: ", args.embed_dim, "mlp_size: ", args.mlp_size, "num_heads: ", args.num_heads, "n_epochs: ", args.n_epochs, "batch_size: ", args.batch_size)
	train_set, val_set, test_set = generate_adni_datasets(datasplit=0.1)

	train_loader = DataLoader(train_set, shuffle=True, batch_size=args.batch_size)
	val_loader = DataLoader(val_set, shuffle=True, batch_size=args.batch_size)
  
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = ViT(img_size=args.img_size, patch_size=args.patch_size, num_transformer_layers=args.num_layers, embedding_dim=args.embed_dim, mlp_size=args.mlp_size, num_heads=args.num_heads).to(device)
	train(model, train_loader=train_loader, val_loader=val_loader, n_epochs=args.n_epochs, version_prefix="vit")

if __name__ == "__main__":
	main()