import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from modules import UNETImproved  

def load_model(model_path, device):
    """
    Loads the trained model from a given path.

    Args:
        - model_path (str): Path to the saved model.
        - device (torch.device): The device to load the model onto.

    Returns:
        - torch.nn.Module: Loaded model.
    """
    model = UNETImproved().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, device, threshold=0.99):
    """
    Runs a prediction on a single image using the provided model.

    Args:
        - model (torch.nn.Module): The trained model for prediction.
        - image_path (str): Path to the image to be predicted.
        - device (torch.device): The device to perform the prediction on.

    Returns:
        - torch.Tensor: The predicted mask.
    """
    # Transform the image (same as used in training)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to the input size expected by your model
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
    
    binary_mask = output > threshold
    return binary_mask.float()

    #return output

def display_prediction(image_path, prediction, actual_mask_path):
    """
    Displays the original image, its prediction, and the actual mask side by side.

    Args:
        - image_path (str): Path to the original image.
        - prediction (torch.Tensor): Predicted mask tensor.
        - actual_mask_path (str): Path to the actual mask image.
    """
    image = Image.open(image_path)
    prediction_image = transforms.ToPILImage()(prediction.squeeze().cpu())
    actual_mask = Image.open(actual_mask_path)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(prediction_image, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(actual_mask, cmap='gray')
    plt.title('Actual Mask')
    plt.axis('off')

    plt.show()

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Paths to the saved model, test image, and actual mask
    model_path = '/content/drive/My Drive/ISIC/Model/model.pth'
    test_image_path = '/content/drive/My Drive/ISIC/IMAGE/ISIC_0013143.jpg'
    mask_image_path = '/content/drive/My Drive/ISIC/MASK/ISIC_0013143_segmentation.png'

    # Load model
    model = load_model(model_path, device)

    # Predict image
    prediction = predict_image(model, test_image_path, device)

    # Display prediction and actual mask
    display_prediction(test_image_path, prediction, mask_image_path)

if __name__ == '__main__':
    main()

