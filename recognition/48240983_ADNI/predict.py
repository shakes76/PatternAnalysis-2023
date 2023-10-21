import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Load a pre-trained ResNet-18 model
model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
model.eval()

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = '/Users/gaurika/pattern/PatternAnalysis-2023/recognition/48240983_ADNI/AD_NC/test/AD/388206_78.jpeg'
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)

# Use the model to make predictions
with torch.no_grad():
    output = model(image)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    predicted_class = 'AD' if probabilities[0, 1] > probabilities[0, 0] else 'NC'

# Plot and save the image
plt.imshow(plt.imread(image_path))
plt.axis('off')
plt.title("Input Image")
plt.savefig('input_image.png')
plt.show()

# Enhanced confidence plot without displaying '0.00%'
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

# Highlight the predicted class
if predicted_class == 'AD':
    bars[0].set_facecolor('red')
else:
    bars[1].set_facecolor('yellow')

# Save the enhanced confidence plot
plt.savefig('confidence_scores.png')
plt.show()

print(f"Predicted Class: {predicted_class}")





