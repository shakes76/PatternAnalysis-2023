import torch
import os
from torchvision import transforms
from PIL import Image
from modules import UNet  # Import your UNet class or module

# Create an instance of your model (replace YourModelClass with your actual model class)
model = UNet(3, 1)  # Instantiate the model with the same architecture as in your training script

# Load the saved model state dictionary
state_dict = torch.load(r'saved model directory here')

# Load the state dictionary into your model
model.load_state_dict(state_dict)

# Set the model in evaluation mode
model.eval()

# Define the test data directory
test_data_dir = r'testing data directory here'

# Define the output directory for saving the predictions
output_dir = r'location of output here'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define a transformation to apply to the test images (make sure it matches your training data preprocessing)
transform = transforms.Compose([transforms.ToTensor()])

# List all image files in the test data directory
test_files = [file for file in os.listdir(test_data_dir) if file.endswith('.jpg')]

# Make predictions for each test image
for idx, file in enumerate(test_files):
    # Load the test image
    image = Image.open(os.path.join(test_data_dir, file))

    # Apply the same transformation used during training
    image = transform(image).unsqueeze(0)  # Add a batch dimension

    # Make predictions
    with torch.no_grad():
        predicted_segmentation = model(image)

    # Convert the predicted segmentation to a format suitable for saving (e.g., numpy array)
    predicted_segmentation = predicted_segmentation[0, 0].cpu().numpy()  # Assuming output has a single channel

    # Save the predicted segmentation to the output directory
    output_filename = os.path.splitext(file)[0] + '_segmentation.png'
    output_path = os.path.join(output_dir, output_filename)
    # Save the predicted segmentation using your preferred method (e.g., PIL, OpenCV)
    # Example using PIL:
    Image.fromarray((predicted_segmentation * 255).astype('uint8')).save(output_path)

    # Print the progress
    #print(f"Processed {idx + 1} out of {len(test_files)} images")

print("Predictions have been saved to the output directory:", output_dir)
