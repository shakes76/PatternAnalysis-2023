import torch
from torchvision import transforms
from modules import get_maskrcnn_model
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the pre-trained model
model = get_maskrcnn_model(num_classes=2)
path_to_saved_model = './Save_Model'
model.load_state_dict(torch.load(path_to_saved_model))
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

path_to_image = './ISIC2018_Task1-2_Test_Input'
image = Image.open(path_to_image).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    predictions = model(image)

# Print results
print(predictions)
