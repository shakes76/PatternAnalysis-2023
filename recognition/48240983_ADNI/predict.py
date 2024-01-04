"""
Created on Thursday 19th Oct
Alzheimer's disease using PyTorch (ViT Transformer)
It first loads the pre-trained ResNet-18 model from PyTorch's model hub and sets up image preprocessing transformations.
The core function of this code is to classify an input image, visualize the image with a confidence plot, and save the results and 
shows the predicted Class.
It does this by loading an image, preprocessing it, and passing it through the pre-trained model to predict the class ('AD'/'NC').

@author: Gaurika Diwan
@ID: s48240983
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
"""
    A simple image classifier using a pre-trained ResNet-18 model to classify an image into  'AD' or 'NC'.

    Args:
        model (torch.nn.Module): A pre-trained ResNet-18 model for image classification.

    Attributes:
        model (torch.nn.Module): The pre-trained model for image classification.
        transform : Image preprocessing transformations.
    """


# Load the pre-trained ResNet-18 model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Define transforms for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
"""
        Classify an image and create an enhanced confidence plot.

        Args:
            image_path (str): The file path to the input image.

        Returns:
            str: The predicted class ('AD' or 'NC').
        """

# Load and preprocess the image data
image_path = '/Users/gaurika/pattern/PatternAnalysis-2023/recognition/48240983_ADNI/AD_NC/test/AD/388206_78.jpeg'
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)

# The model to make predictions
with torch.no_grad():
    output = model(image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = 'AD' if probabilities[0, 1] > probabilities[0, 0] else 'NC'

# Plot and save the image
plt.imshow(plt.imread(image_path))
plt.axis('off')
plt.title("Input")
plt.savefig('input_image.png')
plt.show()

# Accurate confidence plot 
labels = ['AD', 'NC']
scores = [probabilities[0, 1].item() * 100, probabilities[0, 0].item() * 100]
y_pos = range(len(labels))
colors = ['red', 'blue']

fig, ax = plt.subplots()
bars = ax.bar(y_pos, scores, color=colors)
ax.set_xticks(y_pos)
ax.set_xticklabels(labels)
ax.set_ylabel('Confidence (%)')
ax.set_title('Model Confidence Scores')

# Emphasis on  the predicted class
if predicted_class == 'AD':
    bars[0].set_facecolor('red')
else:
    bars[1].set_facecolor('red')

# Save the confidence plot
plt.savefig('training_testing_confidence_scores.png')
plt.show()

print(f"Predicted Class: {predicted_class}")





