import torch
import torchvision.transforms as transforms
from PIL import Image
from modules import UNETImproved
import matplotlib.pyplot as plt

def load_model(model_path):
    """
    Loads a pre-trained UNETImproved model from the specified path.

    Args:
        model_path (str): Path to the saved model state file.

    Returns:
        torch.nn.Module: Loaded UNETImproved model in evaluation mode.
    """
    model = UNETImproved()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    """
    Preprocesses an image from the given path for model prediction.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor.
    """
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)  

def load_mask(mask_path):
    """
    Loads a mask image from the specified path.

    Args:
        mask_path (str): Path to the mask image file.

    Returns:
        torch.Tensor: Mask image tensor.
    """
    mask = Image.open(mask_path)
    transform = transforms.ToTensor()
    return transform(mask)

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)
    return output

def main():
    """
    Main function to load a trained model, preprocess an input image, perform prediction,
    and visualize the results.
    """
     
    model_path = "/content/drive/My Drive/ISIC/Model/model_state.pth" # Path to the saved trained model
    image_path = "/content/drive/My Drive/ISIC/IMAGE/ISIC_0036323.jpg" # Path to the corresponding image input path
    mask_path = "/content/drive/My Drive/ISIC/MASK/ISIC_0036323_segmentation.png"  # Path to the corresponding ground truth mask

    model = load_model(model_path)
    input_tensor = preprocess_image(image_path)
    ground_truth_mask = load_mask(mask_path)
    output = predict(model, input_tensor)

    # Convert tensors to PIL images
    input_image = Image.open(image_path)
    output_image = transforms.ToPILImage()(output.squeeze(0))
    ground_truth_mask_image = transforms.ToPILImage()(ground_truth_mask)

    # Display the images side by side
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(input_image)
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_mask_image, cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(output_image, cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
