import module
import torch
import torchvision.transforms as transforms
from PIL import Image


# Create new model instance
model = module.UNet()

# Load saved weights (pre-trained model)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image input
input_data = Image.open('./predict_image/input_image.jpg')
input_tensor = preprocess(input_data)
input_batch = input_tensor.unsqueeze(0)


# Perform inference
with torch.no_grad():
    output = model(input_batch)
    
