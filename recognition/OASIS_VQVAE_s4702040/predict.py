
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
                                           transforms.Normalize((0.7083, 0.5821, 0.5360), (0.0969, 0.1119, 0.1261)),
                                           transforms.Resize((1024, 672))])

# File path for loading model
filepath = "path to file"

loadedModel = torch.load(filepath)

# Predict mask
print("> Predicting")
loadedModel.eval()
with torch.no_grad():
    # Path to image to be predicted
    testImagePath = "path to file"
    testImage = Image.open(testImagePath).convert('RGB')

    testImage = imageTransform_test(testImage)

    testImage = testImage.to(device)

    # Convert 3D tensor of [c, w, h] to 4D tensor of [n, c, w, h]
    testImage = testImage.unsqueeze(0)

    # Predict mask based in image input
    output = loadedModel(testImage)

    # Convert 4D tensor of [n, c, w, h] to 2D tensor of [w, h]
    output = output.view(1024, 672)

    # Verify dimensions of tensor
    #print(output.size())

    # Load tensor back into cpu and turn into numpy array
    output = output.cpu().numpy()

    # Save predicted mask as png greyscale
    plt.imsave("testPredMask.png", output, cmap="gray")
