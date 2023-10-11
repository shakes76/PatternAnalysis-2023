import os
import cv2

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





def main(txt_folder_path, img_folder_path):
    """
    Iterate through the .txt files in the folder and draw YOLO bounding boxes on the corresponding images.
    """
    for filename in os.listdir(txt_folder_path):
        if filename.endswith(".txt"):
            # Construct paths to the txt file and its corresponding image
            txt_path = os.path.join(txt_folder_path, filename)
            image_path = os.path.join(img_folder_path, filename.replace(".txt", ".jpg"))  # assuming .jpg extension for images

            # Check if the corresponding image exists
            if os.path.exists(image_path):
                draw_yolo_bboxes_on_image(image_path, txt_path)

if __name__ == "__main__":
    TXT_FOLDER_PATH = "/Users/larsmoan/Documents/UQ/COMP3710/PatternAnalysis-2023/lesion_detection_larsmoan/data/ISIC_2017/yolo_labels"
    IMG_FOLDER_PATH = "/Users/larsmoan/Documents/UQ/COMP3710/PatternAnalysis-2023/lesion_detection_larsmoan/data/ISIC_2017/imgs"
    main(TXT_FOLDER_PATH, IMG_FOLDER_PATH)
