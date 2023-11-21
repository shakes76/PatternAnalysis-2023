import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from modules import Perceiver

# Constants and defining to use GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_PATH = './perceiver_model.pth'

# Load the trained model same as training model 
model = Perceiver(
    input_dim = 224 * 224,  
    latent_dim=256, 
    embed_dim=256,
    n_classes=2,
    num_heads=4
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Load and preprocess a sample image (resizing and converting to tensor)
def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Opening image to apply tranformations
    image = Image.open(img_path)
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)
    
    # Predicting by passing image through model using probability
    output = model(image_tensor)
    probabilities = F.softmax(output, dim=1)
    predicted_class = torch.argmax(output, dim=1).item()

    # Display the image and predictions
    plt.imshow(image)
    plt.title(f"Predicted Class: {'NC' if predicted_class == 0 else 'AD'}\nProbability: {probabilities[0][predicted_class]:.4f}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    img_path = input("Enter the path of the image: ")  # Provide path to test images as input
    predict_image(img_path)