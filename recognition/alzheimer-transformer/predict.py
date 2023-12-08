'''
shows example usage of the trained model. 
Prints out any results and provide visualisations where applicable.
'''
from modules import ViT
from dataset import get_alzheimer_dataloader
from train import evaluate
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models.vision_transformer import VisionTransformer
import argparse
from PIL import Image
import torch.nn as nn
from dataset import unnormalise, single_transform

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

    test_acc, test_loss = evaluate(model=model, criterion=nn.CrossEntropyLoss(), device=device, loader=dataloader)

    return most_confident_AD, most_confident_NC, test_acc, test_loss

'''
Displays image with label and confidence score ascociated 
'''
def display_analytics(image, label, confidence):
    plt.imshow(image.permute(1, 2, 0).cpu().numpy())
    plt.title(f'Classified as: {label}, Confidence: {confidence*100:.0f}%')
    plt.axis('off')
    plt.show()
    plt.savefig(f'Class-{label}_Confidence-{confidence*100:.0f}.png')


def main():
    parser = argparse.ArgumentParser(description="Predict using a trained model.")
    parser.add_argument('--model_file', default = None, help='Path to the trained model file')
    parser.add_argument('--image_path', default = None, help='Path to individual image')
    parser.add_argument('--dataset_path', type=str, default='/home/groups/comp3710/ADNI/AD_NC', help='Path to folder containing ADNI Dataset')
    args = parser.parse_args()

    if args.model_file is None:
        raise ValueError("A valid model file path must be provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_file, device)

    # if the user specifies a single image to classify, only classify the image 
    # and return the image along with the class and confidence
    
    if args.image_path:
        tensor_transforms = transforms.Compose([transforms.ToTensor()])
        image = Image.open(args.image_path).convert('RGB')
        pretty_image = tensor_transforms(image)
        output = model(single_transform(image, args.dataset_path + '/train').unsqueeze(0).to(device))
        probs = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_class].item()
        display_analytics(pretty_image, ['AD','NC'][predicted_class], confidence)

    # otherwise, test the model on the test set, returning the test acc and images 
    # which the model was most confident in classifying for each class
    else:
        _, _, test_loader = get_alzheimer_dataloader(batch_size=32, img_size=224, path=args.dataset_path)
        most_confident_AD, most_confident_NC, test_acc, test_loss = confident_classifications(model, test_loader, device)
        display_analytics(unnormalise(most_confident_AD[0], args.dataset_path + '/train'), 'AD', most_confident_AD[1])
        display_analytics(unnormalise(most_confident_NC[0], args.dataset_path + '/train'), 'NC', most_confident_NC[1])
        print(f"Test Accuracy: {test_acc:.4f}\nTest Loss: {test_loss:.4f}")

if __name__ == "__main__":
    main()
        
