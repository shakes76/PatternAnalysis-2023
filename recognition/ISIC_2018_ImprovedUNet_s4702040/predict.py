"""

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
filepath = "filepath\\ImprovedUNet.pt"

loadedModel = torch.load(filepath)

# Predict mask
print("> Predicting")
loadedModel.eval()
with torch.no_grad():
    # Path to image to be predicted
    testImagePath = "filepath\\imageName.jpg"
    testImage = Image.open(testImagePath).convert('RGB')

    testImage = imageTransform_test(testImage)

    testImage = testImage.to(device)

    # Convert 3D tensor of [c, w, h] to 4D tensor of [n, c, w, h]
    testImage = testImage.unsqueeze(0)

    # Predict mask based in image input
    output = loadedModel(testImage)

    # Convert 4D tensor of [n, c, w, h] to 2D tensor of [w, h]
    output = output.view(672, 1024)

    # Verify dimensions of tensor
    #print(output.size())

    # Load tensor back into cpu and turn into numpy array
    output = output.cpu().numpy()

    # Save predicted mask as png greyscale
    plt.imsave("imageName_PredMask.png", output, cmap="gray")
