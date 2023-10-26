'''
shows example usage of the trained model. 
Prints out any results and provide visualisations where applicable.
'''
from modules import ViT
from dataset import get_alzheimer_dataloader
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer
import argparse
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Returns a ViT model given a specifc model parameter file (.pth)
'''
def load_model(model_path, device):
    model = ViT()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

'''
Returns the most confident classifications the model makes for each class
'''
def confident_classifications(model, dataloader, device):
    most_confident_AD = (None, float('-inf'))  # (Image, Confidence Score)
    most_confident_NC = (None, float('-inf'))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
            for i in range(len(labels)):
                if labels[i] == 0 and probs[i][0].item() > most_confident_AD[1]:
                    most_confident_AD = (inputs[i], probs[i][0].item())
                elif labels[i] == 1 and probs[i][1].item() > most_confident_NC[1]:
                    most_confident_NC = (inputs[i], probs[i][1].item())

    return most_confident_AD, most_confident_NC

'''
Displays some analytics and predictions from the ViT given an image. Not yet implemented
'''
def display_analytics(image):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict using a trained model.")
    parser.add_argument('--model_file', default = None, help='Path to the trained model file')
    parser.add_argument('--image_path', default = None, help='Path to individual image')
    parser.add_argument('--dataset_path', type=str, default='dataset/AD_NC', help='Path to folder containing ADNI Dataset')
    args = parser.parse_args()

    if args.model_file is None:
        raise ValueError("A valid model file path must be provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_file, device)

    transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor() 
    ])

    if args.image_path:
        image = transform(Image.open(args.image_path).convert('RGB'))
        display_analytics(image)

    else:
        train_loader, val_loader , test_loader = get_alzheimer_dataloader(batch_size=32, img_size=224, path=args.dataset_path)
        most_confident_AD, most_confident_NC = confident_classifications(model, test_loader, device)
        display_analytics(most_confident_AD[0])
        display_analytics(most_confident_NC[0])

    

    # TODO: Save or display the images using something like matplotlib.
