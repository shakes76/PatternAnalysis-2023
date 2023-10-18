"""
predict.py: This script loads a trained VisionTransformer model and uses it to 
predict the class of a given image. It then visualizes the input image and 
provides a confidence score bar chart.
"""


#importing libraries
import torch
from modules import VisionTransformer
from dataset import test_transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# Constants
MODEL_PATH = '/content/drive/MyDrive/best_model.pth' # Replace with your model path
IMAGE_PATH = '/content/drive/MyDrive/AD/388206_78.jpeg'  # Replace with your image path

def load_model(model_path, device):
    model = VisionTransformer().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def get_true_label(image_path):
    """
    Extract the true label based on the image path.
    Note: This assumes that the class name is part of the directory name.
    """
    return 'AD' if 'AD' in os.path.dirname(image_path) else 'NC'

def predict_image(image_path, model, device, transform):
    """
    Predict the class of the input image.
    Parameters:
        image_path: Path to the image.
        model: Trained model.
        device: CPU or CUDA.
        transform: Image transformations.
    Returns:
        Predicted class and softmax probabilities.
    """
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = 'AD' if probabilities[0, 1] > probabilities[0, 0] else 'NC'
        
    return predicted_class, probabilities

def plot_image(image_path, save_path):
    """
    Display and save the input image.
    
    Parameters:
        image_path: Path to the image.
        save_path: Path to save the displayed image.
    """
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.axis('off')
    plt.title("Input Image")
    plt.savefig(save_path)
    plt.show()

def plot_confidence_scores(probabilities, predicted_class, save_path):
    """
    Plot and save the model's confidence scores for both classes.
    
    Parameters:
        probabilities: Model's softmax output.
        predicted_class: Predicted class ('AD' or 'NC').
        save_path: Path to save the plotted graph.
    """
    labels = ['AD', 'NC']
    scores = [probabilities[0, 1].item()*100, probabilities[0, 0].item()*100]
    y_pos = np.arange(len(labels))
    bars = plt.bar(y_pos, scores, align='center', alpha=0.75, color=['red', 'blue'])
    plt.xticks(y_pos, labels)
    plt.ylabel('Confidence (%)')
    plt.title('Model Confidence Scores')

    # Highlighting the predicted class
    plt.gca().patches[y_pos[labels.index(predicted_class)]].set_facecolor('yellow')

    # Annotating bars with their respective percentages
    for bar in bars:
        yval = bar.get_height()
        plt.annotate(f'{yval:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     xytext=(0, 1), 
                     textcoords="offset points",
                     ha='center', va='bottom')

    # Saving the figure
    plt.savefig(save_path)
    plt.show()

# main function
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(MODEL_PATH, device)

    true_label = get_true_label(IMAGE_PATH)
    plot_image(IMAGE_PATH, '/content/drive/MyDrive/input_image.png')
    
    predicted_class, probabilities = predict_image(IMAGE_PATH, model, device, test_transforms)
    print(f"True Label: {true_label}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Scores - AD: {probabilities[0, 1].item()*100:.2f}%, NC: {probabilities[0, 0].item()*100:.2f}%")
    
    plot_confidence_scores(probabilities, predicted_class, '/content/drive/MyDrive/confidence_scores.png')
