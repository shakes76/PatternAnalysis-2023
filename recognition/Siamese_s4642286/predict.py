# Name: predict.py
# Student: Ethan Pinto (s4642286)
# Description: Shows example usage of trained model.

import torch
from dataset import testloader
from modules import CNN, MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU...')

print(device)

# Load the pre-trained CNN and MLP models
cnn_model = CNN()
cnn_model.load_state_dict(torch.load("./Siamese"))
cnn_model.eval()
cnn_model.to(device)

mlp_model = MLP()
mlp_model.load_state_dict(torch.load("path_to_mlp_model.pth"))
mlp_model.eval()
mlp_model.to(device)

# Define a function to calculate accuracy
def calculate_accuracy(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            feature_vector = cnn_model(images)
            outputs = mlp_model(feature_vector)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# Calculate accuracy on the test dataloader
test_accuracy = calculate_accuracy(mlp_model, testloader)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')


# Need a minimum accuracy of 0.8 on the test set.


