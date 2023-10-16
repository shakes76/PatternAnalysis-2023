""" Showing example usage of trained model. """
import torch
from tqdm import tqdm

def test(model, device, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    
    # Test loop
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            
            with torch.cuda.amp.autocast():
                y_hat = model(x)
                loss = criterion(y_hat, y)

            test_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        test_loss /= len(test_loader)  # Calculate average test loss
        accuracy = 100 * correct / total  # Calculate accuracy
        
        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {accuracy:.2f}%")