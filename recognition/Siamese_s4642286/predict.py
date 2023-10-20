# Name: predict.py
# Student: Ethan Pinto (s4642286)
# Description: Shows example usage of trained model.

import torch
from torchvision import transforms
from PIL import Image
import sys
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


# Load and preprocess the image from the command-line argument
image_path = sys.argv[1]  # Assuming the image path is provided as a command-line argument
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image to 128x128
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

image = Image.open(image_path).convert("RGB")  # Load the image and ensure it's in RGB mode
image = transform(image)  # Apply the transformation

# Pass the image through the Siamese Network
with torch.no_grad():
    feature_vector = siamese_net.forward_once(image.unsqueeze(0).to(device))  # Ensure input has the expected batch dimension

# Pass the feature vector through the MLP for classification
output = mlp_model(feature_vector)

# Example threshold for binary classification (adjust as needed)
threshold = 0.5
predicted_class = 1 if output.item() > threshold else 0

# Print the predicted class and the raw output
print(f"Predicted Class: {predicted_class}")
print(f"Raw Output: {output.item()}")


# # Define a function to calculate accuracy
# def calculate_accuracy(dataloader):
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for data in dataloader:
#             images, labels = data
#             images, labels = images.to(device), labels.to(device)
            
#             feature_vector = siamese_net.forward_once(images)
#             outputs = mlp_model(feature_vector)
            
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
    
#     return correct / total

# # Calculate accuracy on the test dataloader
# test_accuracy = calculate_accuracy(testloader)

# # specify a path to an image to send, show predict label vs actual label + probability.

# print(f'Test Accuracy: {test_accuracy * 100:.2f}%') # Need a minimum accuracy of 0.8 on the test set.
