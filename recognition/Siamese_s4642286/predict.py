# Name: predict.py
# Student: Ethan Pinto (s4642286)
# Description: Shows example usage of trained model.

import torch
from dataset import testloader
from modules import SiameseNetwork, MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('CUDA not available. Running on CPU...')

print(device)

##########################################################################################
################                    Testing Model                           ##############
##########################################################################################

# Load the pre-trained Siamese Network and MLP models for inference
siamese_net = SiameseNetwork()
siamese_net.load_state_dict(torch.load("PatternAnalysis-2023/recognition/Siamese_s4642286/Siamese/siamese_model.pt"))
siamese_net.to(device)
siamese_net.eval()

mlp_model = MLP(128, 64, 1)
mlp_model.load_state_dict(torch.load("PatternAnalysis-2023/recognition/Siamese_s4642286/Classifier/classifier.pt"))
mlp_model.to(device)
mlp_model.eval()

# Define a function to calculate accuracy
def calculate_accuracy(dataloader):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            
            feature_vector = siamese_net.forward_once(images)
            outputs = mlp_model(feature_vector)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# Calculate accuracy on the test dataloader
test_accuracy = calculate_accuracy(testloader)

# specify a path to an image to send, show predict label vs actual label + probability.

print(f'Test Accuracy: {test_accuracy * 100:.2f}%') # Need a minimum accuracy of 0.8 on the test set.
