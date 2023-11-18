"""
This file is to be used after train.py has been run since it relies on loading a trained model

NOTES:
- Loads a single image and so will output a single segmentation mask prediction
- Must have modules.py within the same folder as predict.py or else loading the model will fail
"""

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

imageTransform_test = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((672, 1024))])

# File path for loading model
filepath = "path to file\\ImprovedUNet.pt"

loadedModel = torch.load(filepath) #load model from file path

# Predict mask
print("> Predicting")
loadedModel.eval() #set model to evaluation mode
with torch.no_grad():
    # Path to image which will have its segmentation mask predicted
    testImagePath = "path to file\\ISIC2018\\ISIC2018_Task1-2_Test_Input\\ISIC_0016911.jpg"
    testImage = Image.open(testImagePath).convert('RGB') #convert image to RGB values

    testImage = imageTransform_test(testImage) #apply transform to image

    testImage = testImage.to(device) #send image to GPU

    # Convert 3D tensor of [c, h, w] to 4D tensor of [n, c, h, w]
    testImage = testImage.unsqueeze(0)

    # Predict mask based in image input
    output = loadedModel(testImage)

    # Convert 4D tensor of [n, c, h, w] to 2D tensor of [h, w]
    output = output.view(672, 1024)

    # Verify dimensions of tensor
    #print(output.size())

    # Load tensor back into cpu and turn into numpy array
    output = output.cpu().numpy()

    # Save predicted mask as png greyscale in folder where predict.py is located
    plt.imsave("ISIC_0016911_PredMask.png", output, cmap="gray")
