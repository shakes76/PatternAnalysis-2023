import torch
import os
from torchvision import transforms
from PIL import Image
from modules import UNet  # Import your UNet class or module

# Load the saved model
model = UNet(3, 1)  # Instantiate the model (make sure to use the same architecture as in your training script)
model.load_state_dict(torch.load(''))
model.eval()

# Define the test data directory
test_data_dir = r'C:\Users\sam\Downloads\ISIC2018_Task1-2_SegmentationData_x2\ISIC2018_Task1-2_Test_Input'

# Define the output directory for saving the predictions
output_dir = r'C:\Users\sam\Downloads\ISIC2018_Task1-2_SegmentationData_x2\predictions'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a transformation to apply to the test images (make sure it matches your training data preprocessing)
transform = transforms.Compose([transforms.ToTensor()])

# List all image files in the test data directory
test_files = [file for file in os.listdir(test_data_dir) if file.endswith('.jpg')]

# Make predictions for each test image
for file in test_files:
    # Load the test image
    image = Image.open(os.path.join(test_data_dir, file))

    # Apply the same transformation used during training
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Make predictions
    with torch.no_grad():
        predicted_segmentation = model(image)

    # Convert the predicted segmentation to a format suitable for saving (e.g., numpy array)
    predicted_segmentation = predicted_segmentation[0].cpu().numpy()

    # Save the predicted segmentation to the output directory
    output_filename = os.path.splitext(file)[0] + '_segmentation.png'
    output_path = os.path.join(output_dir, output_filename)
    # Save the predicted segmentation using your preferred method (e.g., PIL, OpenCV)
    # Example using PIL:
    Image.fromarray((predicted_segmentation * 255).astype('uint8')).save(output_path)

print("Predictions have been saved to the output directory:", output_dir)
