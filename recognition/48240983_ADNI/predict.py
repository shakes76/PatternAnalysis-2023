import os
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from modules import VisionModel  # Import your VisionModel module here

# Constants
MODEL_PATH = '/model.pth'  # Replace with your model path
IMAGE_PATH = '/recognition/48240983_ADNI/AD_NC/test/AD/388206_79.jpeg'  # Replace with your image path

def load_model(model_path, device):
    model = VisionModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def get_true_label(image_path):
    return 'AD' if 'AD' in os.path.dirname(image_path) else 'NC'

def predict_image(image_path, model, device, transform):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_class = 'AD' if probabilities[0, 1] > probabilities[0, 0] else 'NC'

    return predicted_class, probabilities

def plot_image(image_path, save_path):
    image = Image.open(image_path).convert('RGB')
    plt.imshow(image)
    plt.axis('off')
    plt.title("Input Image")
    plt.savefig(save_path)
    plt.close()  # Close the figure

def plot_confidence_scores(probabilities, predicted_class, save_path):
    labels = ['AD', 'NC']
    scores = [probabilities[0, 1].item() * 100, probabilities[0, 0].item() * 100]
    y_pos = np.arange(len(labels))

    bars = plt.bar(y_pos, scores, align='center', alpha=0.75, color=['red', 'blue'])
    plt.xticks(y_pos, labels)
    plt.ylabel('Confidence (%)')
    plt.title('Model Confidence Scores')

    plt.gca().patches[y_pos[labels.index(predicted_class)]].set_facecolor('yellow')

    for bar in bars:
        yval = bar.get_height()
        plt.annotate(f'{yval:.2f}%',
                     xy=(bar.get_x() + bar.get_width() / 2, yval),
                     textcoords="offset points",
                     ha='center', va='bottom')

    plt.savefig(save_path)

    plt.show()

# ... (previous code) ...

if __name__ == '__main':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = load_model(MODEL_PATH, device)
    true_label = get_true_label(IMAGE_PATH)
    plot_image(IMAGE_PATH, 'recognition/48240983_ADNI/AD_NC')

    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    predicted_class, probabilities = predict_image(IMAGE_PATH, model, device, transform)

    print(f"True Label: {true_label}")
    print(f"Predicted Class: {predicted_class}")
    print(f"Confidence Scores - AD: {probabilities[0, 1].item() * 100:.2f}%, NC: {probabilities[0, 0].item() * 100:.2f}%")

    # plt.ion()  # Deactivate interactive mode
    plot_confidence_scores(probabilities, predicted_class, 'confidence_scores.png')
    plt.ioff()  # Deactivate interactive mode
    plt.show()  # Explicitly display the plot
