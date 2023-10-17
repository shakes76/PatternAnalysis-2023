import torch
from dataset import get_loaders
from module import create_model

"""
Loads the best model and evaluates it on the test set.
"""

def test(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total_samples = len(data_loader.dataset)

    with torch.no_grad():
        for batch, labels in data_loader:
            batch, labels = batch.to(device), labels.to(device)
            prediction = model(batch)
            correct_predictions += torch.sum(torch.argmax(prediction, dim=1) == labels)

    accuracy = correct_predictions / total_samples

    return accuracy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_data = get_loaders()
    
    model, _, _, _ = create_model(
        input_shape=(256, 256),            # Reduce input image shape
        latent_dim=8,                   # Smaller latent space dimension
        embed_dim=16,                     # Smaller image patch dimension
        attention_mlp_dim=16,            # Smaller dimension for cross-attention's feedforward network
        transformer_mlp_dim=16,          # Smaller dimension for the latent transformer's feedforward network
        transformer_heads=4,              # Fewer attention heads for the latent transformer
        dropout=0.1,                      # Reduce dropout for lower memory usage
        transformer_layers=4,            # Fewer layers in the latent transformer
        n_blocks=4,                       # Fewer Perceiver blocks
        n_classes=2,                      # Number of target classes (binary classification)
        batch_size=32,                     # Further reduce batch size to save memory
        lr=0.005,                        # Smaller learning rate for stability
    )

    model = model.to(device)
    
    best_model_state = torch.load('saved/best_model.pth')
    model.load_state_dict(best_model_state['model_state_dict'])


    test_acc = test(model, test_data, device)
    print(f"Test accuracy: {test_acc}")
    

if  __name__ == "__main__":
    main()
