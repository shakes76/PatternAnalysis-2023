import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import zipfile
import gdown
import signal
import sys
from tqdm import tqdm
import shutil


def visualize_labels(dataset_folder):
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            txt_path = os.path.join(dataset_folder, filename)
            image_path = os.path.join(dataset_folder, filename.replace(".txt", ".jpg"))
            if os.path.exists(image_path):
                draw_yolo_bboxes_on_image(image_path, txt_path)
            else:
                print("Could not find the corresponding image for", txt_path)


#Removes all superpixel images from a folder
def remove_superpixels(folder):
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for file in files:
        if file.endswith('superpixels.png'):
            os.remove(os.path.join(folder, file))


#Computes the bounding box of a binary mask, not normalized
def compute_bbx(binary_mask):
    # Find the coordinates of non-zero pixels
    rows, cols = np.where(binary_mask == 255)

    # Get the minimum and maximum coordinates
    x_min, x_max = np.min(cols), np.max(cols)
    y_min, y_max = np.min(rows), np.max(rows)

    # Compute the center, width, and height
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height

#Based on the binary / segmentaiton images, create bounding boxes, normalize them to YOLO format and fetch the class from the csv file
def create_yolo_labels(seg_folder_path, class_csv_file, dst_folder_path):
    labels = pd.read_csv(class_csv_file, index_col=0)
    
    for filename in tqdm(os.listdir(seg_folder_path), desc="Processing images"):
        if filename.endswith(".png"):
            image_id = filename[:12]

            # Determine the class based on the dataframe
            class_row = labels.loc[image_id]
            if class_row['melanoma'] == 1.0:
                class_id = 0
            elif class_row['seborrheic_keratosis'] == 1.0:
                class_id = 1
            else:
                class_id = 2  # Assuming a third class for 'other' or 'nevi' based on the given example

            # Load the binary image
            image_path = os.path.join(seg_folder_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            # Compute the bounding box
            x_center, y_center, width, height = compute_bbx(image)
            
            # Normalize the coordinates -> YOLO format
            x_center /= image.shape[1]
            y_center /= image.shape[0]
            width /= image.shape[1]
            height /= image.shape[0]

            # Write the YOLO bbx with class to a .txt file
            output_path = os.path.join(dst_folder_path, filename.replace("_segmentation.png", ".txt"))  
            with open(output_path, "w") as file:
                file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")



def draw_yolo_bboxes_on_image(image_path, txt_path):
    image = cv2.imread(image_path)

    # Extracting file ID from image_path
    file_id = os.path.splitext(os.path.basename(image_path))[0]

    #color mapping based on class_id
    class_colors = {
        0: (0, 255, 0),  # Green for Melanoma
        1: (0, 0, 255)   # Red for Seborrheic Keratosis
    }

    #Resizing factors for displaying
    resize_factor_x = 480 / image.shape[1]
    resize_factor_y = 480 / image.shape[0]

    resized_image = cv2.resize(image, (480, 480))

    # Display the file ID on top of the image
    cv2.putText(resized_image, file_id, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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

            cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(resized_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Image with Bounding Boxes and Labels', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def visualize_labels(dataset_folder):
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".txt"):
            txt_path = os.path.join(dataset_folder, filename)
            image_path = os.path.join(dataset_folder, filename.replace(".txt", ".jpg"))
            if os.path.exists(image_path):
                draw_yolo_bboxes_on_image(image_path, txt_path)
            else:
                print("Could not find the corresponding image for", txt_path)


def downsample_images(input_folder_path, output_folder, downsample_factor):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for filename in tqdm(os.listdir(input_folder_path), desc="Downsampling images"):
        if filename.endswith(".jpg"):
            filepath = os.path.join(input_folder_path, filename)
            
            try:
                # Load the image
                img = cv2.imread(filepath)
                
                # Check if the image is loaded correctly
                if img is None:
                    print(f"Failed to load {filename}. Skipping...")
                    continue
                
                # Calculate the new size
                new_size = (img.shape[1] // downsample_factor, img.shape[0] // downsample_factor)
                
                # Resize the image
                resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
                
                # Save the downsampled image to the output folder
                cv2.imwrite(os.path.join(output_folder, filename), resized_img)
                    
            except Exception as e:
                print(f"Error processing {filename}: {e}")



def copy_txt_files(source_folder, destination_folder):
    # Ensure destination directory exists; if not, create it
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for filename in tqdm(os.listdir(source_folder), desc="Copying label files"):
        if filename.endswith('.txt'):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            
            shutil.copy2(source_path, destination_path)


#Used to download and unzip compressed dataset from google drive if the dataset folder doesn't exist
def download_dataset(dataset_folder, download_url="https://drive.google.com/uc?id=1YI3pwanX35i7NCIxKnfXBozXiyQZcGbL"):
    if not os.path.exists(dataset_folder):
        print(f"Folder {dataset_folder} does not exist. Downloading the zip file...")
        
        # Download zip file using gdown
        output = "temp_download.zip"
        gdown.download(download_url, output, quiet=False)
        
        # Unzip the downloaded file
        print("Unzipping the downloaded file...")
        with zipfile.ZipFile(output, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)
        
        # Delete the downloaded zip file
        os.remove(output)
        print(f"Files have been extracted to {dataset_folder}")
    else:
        print(f"Folder {dataset_folder} already exists.")

if __name__ == "__main__":
    #Download the preprocessed version of ISIC2017 that has YOLO labels and correct structure
    path_to_dset = "/home/Student/s4827064/PatternAnalysis-2023/lesion_detection_larsmoan/data/"
    download_dataset(path_to_dset)

    #Download pretrained YOLOV7 weights (COCO) for transfer learning purposes
    weights_url = "https://drive.google.com/uc?id=1mAu29ZlOTn3csjnZ5fmro10kY3XxAddC"
    yolov7_weights_out = "yolov7_training.pt"
    gdown.download(weights_url, yolov7_weights_out, quiet=False)
   
    