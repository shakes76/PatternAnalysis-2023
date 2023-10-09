import module
import utils
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create new model instance
model = module.UNet()

# Load saved weights (pre-trained model)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess image input
input_data = Image.open('./predict_image/input_image.png')
input_tensor = preprocess(input_data)
input_batch = input_tensor.unsqueeze(0)

# Save image for predict.py
def save_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),
        transforms.Lambda(lambda t: t * 255),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage()
    ])
     
    # Shape image being saved so its a single brain
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    
    # save image here  
    image_path = './denoised_image.png'
    img = reverse_transforms(image)
    img.save(image_path)

# Perform inference
with torch.no_grad():
    t = torch.randint(0, utils.T, (utils.BATCH_SIZE,), device=device).long()
    output = model(input_batch, t)
  
    img_size = utils.IMAGE_SIZE
    img = torch.randn((1, 1, img_size, img_size), device=device)
    num_images = 1
    stepsize = int(utils.T/num_images)

    for i in range(0, utils.T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = utils.sample_timestep(model, img, t)
        # Maintain natural range of distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            save_tensor_image(img.detach().cpu())
