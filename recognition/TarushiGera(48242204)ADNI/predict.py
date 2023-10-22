"""
Swin Transformer Model Architecture Based Topic Recognition for Alzheimer's Disease Classification
Name: Tarushi Gera
Student ID: 48242204
This script allows you to utilize the model to identify whether the input image is AD or NC. 
The image is preprocessed and passed through the architecture, and the result is displayed with the input image.
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import preprocess_image # Import dataset class from dataset.py
from modules import SwinTransformer # Import Swin transformer model from modules.py


class Tester:
    def __init__(self, args):
        self.image_path = args.image_path
        self.config_path = args.config_path
        
        with open(self.config_path, "r") as config_file:
            config = yaml.safe_load(config_file)
        # Read and set configuration parameters
        self.device = config['training']['device']
        self.num_classes = config['model']['num_classes']     
        self.image_size = config['training']['image_size']
        
        self.save_dir = config['training']['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),  # Resize images to a fixed size
            transforms.ToTensor(),           # Convert to tensor
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize pixel values
        ])  
    
        # Loading Model
        self.model = SwinTransformer(
            image_size=self.image_size[0], 
            num_classes=self.num_classes)
        
        # Loading checkpoint (only for testing phase)
        if config['testing']['model_path']:
            checkpoint = torch.load(config['testing']['model_path'])
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        
        
    def load_data(self, image_path):
        image = cv2.imread(image_path)
        image = preprocess_image(image)
        if self.transform:
            image = self.transform(image)
        
        return image.unsqueeze(0).to(self.device)
    
    def predict(self):
        inputs = self.load_data(self.image_path)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        
        image_np = inputs.cpu().numpy()[0]
        image_np = np.transpose(image_np, (1, 2, 0))

        # Display the image using matplotlib
        plt.imshow(image_np, cmap='gray')  # Use 'gray' colormap for grayscale images
        plt.axis('off')  # Turn off axis labels
        plt.show()
        
        if predicted.cpu().numpy()[0] == 0:
            print("Alzheimers Disease")
        else:
            print("No Disease")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alzheimer Disease")
    parser.add_argument("--image_path", type=str, help="Path to the input image")
    parser.add_argument("--config_path", type=str, help="Path to the trained model", default='config.yaml')
    args = parser.parse_args()
    tester = Tester(args)
    tester.predict()
