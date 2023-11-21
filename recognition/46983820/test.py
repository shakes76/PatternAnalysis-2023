import torch
from dataset import get_loaders
from module import create_model

"""
Loads the best model and evaluates it on the test set.
"""

def test(model, data_loader, device):
    """
    Test the model on the test set.
    
    Args:
        model (torch.nn.Module): The model to test.
        data_loader (torch.utils.data.DataLoader): The data loader for the test set
        device (torch.device): The device to use.
    
    Returns:
        accuracy (float): The test accuracy.
    """
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
    
    model, optimizer, criterion, scheduler = create_model(
        input_shape=(256, 256),
        latent_dim=32, # Increase latent space dimension for more representational capacity
        embed_dim=32,
        attention_mlp_dim=32,
        transformer_mlp_dim=32,
        transformer_heads=4, # Use more attention heads for enhanced feature capturing
        dropout=0.1,
        transformer_layers=4,
        n_blocks=4, # more perceiver blocks for improved Representation Learning
        n_classes=2,
        lr=0.003,
    )

    model = model.to(device)
    
    best_model_state = torch.load('saved/best_model.pth')
    model.load_state_dict(best_model_state['model_state_dict'])


    test_acc = test(model, test_data, device)
    print(f"Test accuracy: {test_acc}")
    

if  __name__ == "__main__":
    main()
