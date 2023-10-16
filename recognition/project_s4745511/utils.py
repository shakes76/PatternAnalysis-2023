#The following codes were tried on Google colab

#1. 
##Importing the Zip file from the url: https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download
import requests

# URL of the file to download
url = "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"

# Define the local file path where you want to save the downloaded file
local_file_path = "/content/downloaded_file.zip"  # You can change the name and location if needed

# Send an HTTP GET request to the URL to download the file
response = requests.get(url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Write the downloaded content to the local file
    with open(local_file_path, 'wb') as file:
        file.write(response.content)
    print("File downloaded successfully.")
else:
    print(f"Failed to download the file. Status code: {response.status_code}")


#2.
#The downloaded zip file is extracted
import zipfile

# Path to the downloaded zip file
# zip_file_path = "/content/downloaded_file.zip"  # Replace with the actual path of your downloaded zip file

# Directory where you want to extract the contents
extracted_dir_path = "/content/extracted"  # Replace with your desired extraction directory

# Create the extraction directory if it doesn't exist
import os
os.makedirs(extracted_dir_path, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(local_file_path, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir_path)

# List the extracted files (optional)
extracted_files = os.listdir(extracted_dir_path)
print("Extracted Files:")
print(extracted_files)


#3.
#Understanding the ADNI dataset

#Number of images in train and test
import os
import cv2  # You may need to install OpenCV using !pip install opencv-python
import numpy as np

# Directory containing the extracted images
base_dir = "/content/extracted"  # Change this to your extracted image directory

# Subdirectories for train and test sets
train_dir = os.path.join(base_dir, "AD_NC", "train")
test_dir = os.path.join(base_dir, "AD_NC", "test")

# Function to load images from a directory
def load_images_from_directory(directory):
    image_data = []
    labels = []

    for sub_dir in os.listdir(directory):
        label = sub_dir  # Subdirectory name is used as the label
        sub_dir_path = os.path.join(directory, sub_dir)

        if os.path.isdir(sub_dir_path):
            for filename in os.listdir(sub_dir_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(sub_dir_path, filename)
                    image = cv2.imread(image_path)  # Load the image using OpenCV
                    image_data.append(image)
                    labels.append(label)

    return np.array(image_data), np.array(labels)

# Load train and test images and labels
train_images, train_labels = load_images_from_directory(train_dir)
test_images, test_labels = load_images_from_directory(test_dir)

# Print the shapes of the loaded data (number of images and their dimensions)
print("Train Images Shape:", train_images.shape)
print("Train Labels Shape:", train_labels.shape)
print("Test Images Shape:", test_images.shape)
print("Test Labels Shape:", test_labels.shape)
#output
# Train Images Shape: (21520, 240, 256, 3)
# Train Labels Shape: (21520,)
# Test Images Shape: (9000, 240, 256, 3)
# Test Labels Shape: (9000,)

#4. Printing the images in AD and NC
import matplotlib.pyplot as plt
import random

# Display a random training image
random_index = random.randint(0, len(train_images) - 1)
plt.figure(figsize=(5, 5))
plt.imshow(train_images[random_index])
plt.title(f"Train Label: {train_labels[random_index]}")
plt.axis('off')
plt.show()

# Display a random testing image
random_index = random.randint(0, len(test_images) - 1)
plt.figure(figsize=(5, 5))
plt.imshow(test_images[random_index])
plt.title(f"Test Label: {test_labels[random_index]}")
plt.axis('off')
plt.show()

import matplotlib.pyplot as plt
import random

# Define the categories (AD and NC)
categories = ["AD", "NC"]

# Function to display a random image from a specified category and dataset
def display_random_image(category, dataset):
    category_dir = os.path.join(dataset, category)
    random_image_path = os.path.join(category_dir, random.choice(os.listdir(category_dir)))
    random_image = cv2.imread(random_image_path)
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(random_image, cv2.COLOR_BGR2RGB))
    plt.title(f"{category} Image")
    plt.axis('off')
    plt.show()

# Display a random AD and NC image from the training dataset
for category in categories:
    display_random_image(category, train_dir)

# Display a random AD and NC image from the testing dataset
for category in categories:
    display_random_image(category, test_dir)
