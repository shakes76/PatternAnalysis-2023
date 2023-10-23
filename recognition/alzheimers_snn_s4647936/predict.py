import torch
from modules import SiameseNetwork
from torchvision import transforms
from PIL import Image

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
model.eval()

# Load a sample image (or two images for Siamese)
transform = transforms.Compose([
    transforms.Resize((256, 240)),
    transforms.ToTensor(),
])
image1 = Image.open("path_to_sample_image_1.jpg")
image2 = Image.open("path_to_sample_image_2.jpg")
tensor1 = transform(image1).unsqueeze(0).to(device) 
tensor2 = transform(image2).unsqueeze(0).to(device)

# Predict
output1, output2 = model(tensor1, tensor2)
print(output1, output2) 
