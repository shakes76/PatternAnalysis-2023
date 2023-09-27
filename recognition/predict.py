import torch
from torchvision import transforms
from modules import get_maskrcnn_model
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load the pre-trained model
model = get_maskrcnn_model(num_classes=2)
path_to_saved_model = 'C:\\Users\\yangj\\Desktop\\COMP3710_Project\\Save_Model'
model.load_state_dict(torch.load(path_to_saved_model))
model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load an image for prediction
path_to_image = 'C:\\Users\\yangj\\Desktop\\COMP3710 Project Test1\\ISIC-2017_Validation_DataData'
image = Image.open(path_to_image).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    predictions = model(image)

# Print results
print(predictions)
