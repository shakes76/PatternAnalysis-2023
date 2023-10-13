import os
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
# , map_location=torch.device('cpu')
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((utils.IMAGE_SIZE, utils.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Load image inputs into a list
image_paths = []
for subdir, _, files in os.walk('./predict_image'):
    for file in files:
        if file.endswith(".png"):
            image_paths.append(os.path.join(subdir, file))

input_datas = [Image.open(i) for i in image_paths]

# Save image for predict.py
def save_tensor_image(image, counter):
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
    image_path = f'./predict_output/generated_image_{counter}.png'
    img = reverse_transforms(image)
    img.save(image_path)

# Perform inference
with torch.no_grad():
    counter = 0
    t = torch.randint(0, utils.T, (utils.BATCH_SIZE,), device=device).long()
    
    for input_data in input_datas:
        # Preprocess input data
        input_tensor = preprocess(input_data)
        input_batch = input_tensor.unsqueeze(0)

        output = model(input_batch, t)
        
        img_size = utils.IMAGE_SIZE
        
        # Generate random noise sample image
        img = torch.randn((1, 1, img_size, img_size), device=device) 
        num_images = 1
        stepsize = int(utils.T/num_images)

        for i in range(0, utils.T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            img = utils.sample_timestep(model, img, t)
            # Maintain natural range of distribution
            img = torch.clamp(img, -1.0, 1.0)
            if i % stepsize == 0:
                save_tensor_image(img.detach().cpu(), counter)
                counter += 1
