import os
import cv2
import zipfile
import gdown

def draw_yolo_bboxes_on_image(image_path, txt_path):
    """
    Draw YOLO bounding boxes from a .txt file on an image, display labels, and show the image in a size of 240x240 pixels.
    """
    # Load the image
    image = cv2.imread(image_path)

    # Define color mapping based on class_id
    class_colors = {
        0: (0, 255, 0),  # Green for Melanoma
        1: (0, 0, 255)   # Red for Seborrheic Keratosis
    }

    # Compute the resizing factors
    resize_factor_x = 240 / image.shape[1]
    resize_factor_y = 240 / image.shape[0]

    # Resize the image
    resized_image = cv2.resize(image, (240, 240))
    
    # Read bounding box data from the txt file
    with open(txt_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())

            # Denormalize the coordinates
            x_center *= image.shape[1]
            y_center *= image.shape[0]
            width *= image.shape[1]
            height *= image.shape[0]

            # Adjust the bounding box coordinates based on the resizing factors
            x_center *= resize_factor_x
            y_center *= resize_factor_y
            width *= resize_factor_x
            height *= resize_factor_y

            # Convert to top-left x, y coordinates
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)

            # Define the label and color based on class_id
            label = "Melanoma" if class_id == 0 else "Seborrheic Keratosis" if class_id == 1 else "Unknown"
            color = class_colors.get(class_id, (255, 255, 255))  # Default color is white

            # Draw the bounding box on the resized image with the defined color
            cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), color, 2)

            # Overlay the label on the image with the same color
            cv2.putText(resized_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resized image with bounding boxes and labels
    cv2.imshow('Image with Bounding Boxes and Labels', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def download_and_unzip(folder_path, download_url):
    """
    Checks if a folder exists at the specified path. 
    If not, downloads a zip file from the provided URL using gdown and unzips it.
    
    Parameters:
    - folder_path: Path to the folder you want to check.
    - download_url: The gdown URL of the zip file to download if the folder doesn't exist.
    """

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist. Downloading the zip file...")
        
        # Download zip file using gdown
        output = "temp_download.zip"
        gdown.download(download_url, output, quiet=False)
        
        # Unzip the downloaded file
        print("Unzipping the downloaded file...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(folder_path)
        
        # Delete the downloaded zip file
        os.remove(output)
        print(f"Files have been extracted to {folder_path}")
    else:
        print(f"Folder {folder_path} already exists.")




def visualize_labels(dataset_folder):
    """
    Iterate through the .txt files in the folder and draw YOLO bounding boxes on the corresponding images.
    """
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            # Construct paths to the txt file and its corresponding image
            txt_path = os.path.join(dataset_folder, filename)
            image_path = os.path.join(dataset_folder, filename.replace(".txt", ".jpg"))  # assuming .jpg extension for images

            # Check if the corresponding image exists
            if os.path.exists(image_path):
                draw_yolo_bboxes_on_image(image_path, txt_path)

import os
from PIL import Image

def get_image_dimensions(image_path):
    """Return the dimensions of the image."""
    with Image.open(image_path) as img:
        return img.size

def get_all_image_dimensions_in_folder(folder_path, valid_extensions={"jpg", "jpeg", "png"}):
    """Return a list of dimensions of all images in the folder."""
    
    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith(tuple(valid_extensions))]

    all_dimensions = []

    for file_name in all_files:
        file_path = os.path.join(folder_path, file_name)
        try:
            dimensions = get_image_dimensions(file_path)
            all_dimensions.append(dimensions)
        except Exception as e:
            # This means the file is probably not an image or there's an issue reading it
            print(f"Could not process {file_name}. Error: {e}")

    return all_dimensions

def analyze_image_dimensions(folder_path):
    dimensions = get_all_image_dimensions_in_folder(folder_path)

    if not dimensions:
        print("No valid images found in the specified folder.")
        return

    widths, heights = zip(*dimensions)

    # Calculating statistics
    max_width, max_height = max(widths), max(heights)
    min_width, min_height = min(widths), min(heights)
    mean_width, mean_height = sum(widths) / len(widths), sum(heights) / len(heights)

    # Since mean and average are the same, we'll just display the mean
    print(f"Max Width: {max_width}, Max Height: {max_height}")
    print(f"Min Width: {min_width}, Min Height: {min_height}")
    print(f"Mean/Average Width: {mean_width:.2f}, Mean/Average Height: {mean_height:.2f}")




if __name__ == "__main__":
    folder = "/Users/larsmoan/Documents/UQ/COMP3710/PatternAnalysis-2023/lesion_detection_larsmoan/data/ISIC_2017_downsampled/train_downsampled"
    #visualize_labels(folder)

    analyze_image_dimensions(folder)
    

    """ dataset_folder = "path/to/check/folder"
    url = "https://drive.google.com/uc?id=19HcgRBuXyhzxsukE2jHPnBkC0hEk_aMJ" """

    #download_and_unzip(dataset_folder, url)

