import os
import argparse
import torch
from torchvision import transforms
from PIL import Image
from modules import ConvolutionalVisionTransformer, CViTConfig
from dataset import load_test_data,configuration

def predict_image(image_path,model_path):
    """
    Predict the class of an image using a pre-trained CViT model. An image is read and preprocessed from specified path, before feeding it into CViT model. The model 
    processes the image to determine the class. Function returns the class index predicted by the model.
    """
    # Define a transform to preprocess the test image (adjust as needed)
    transform = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
    ])

    # Load the model and set it to evaluation mode
    config_params_dict=configuration()
    config = CViTConfig(config_params_dict)
    model = ConvolutionalVisionTransformer(config)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Read and preprocess the test image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    with torch.no_grad():
        output = model(image)

    # Process the model's output to get the predicted class
    _, predicted_class = torch.max(output, 1)

    return predicted_class.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Prediction with CvT Model")
    parser.add_argument("image_path", type=str, help="Path to the image file for prediction")
    args = parser.parse_args()
    model_save_path = "models.pth"
    predicted_class = predict_image(args.image_path,model_save_path)
    print(f"Predicted Class: {predicted_class}")
