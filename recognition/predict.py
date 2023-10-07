import torch
from torchvision import transforms
from modules import get_maskrcnn_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set up the device for GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize and load the pre-trained model
model_path = './Save_Model'
model = get_maskrcnn_model(num_classes=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define the image transform pipeline
def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    return preprocess(img).unsqueeze(0)

# Load and preprocess the image
image_path = './ISIC2018_Task1-2_Test_Input'
input_image = preprocess_image(image_path).to(device)

# Perform inference
with torch.no_grad():
    prediction = model(input_image)

def visualize_predictions(image_tensor, predictions):
    image_arr = image_tensor[0].cpu().numpy().transpose((1, 2, 0))
    plt.imshow(image_arr)
    for box in predictions[0]['boxes']:
        plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red'))
    plt.show()

visualize_predictions(input_image, prediction)

# Display prediction results
print(prediction)
