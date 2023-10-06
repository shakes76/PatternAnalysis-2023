import os
import cv2
import numpy as np


def load_images(folder_path, target_size=(256, 256), extensions=[".jpg", ".png"]):
    """

    load the images and rezise them to target size

    :param folder_path: path to folder containing images
    :param target_size: size of image to resize
    :param extensions: list of file extensions to consider as images
    :return: numpy array of resized images
    """

    # Check if the folder_path exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path '{folder_path}' does not exist.")

    resized_images = []

    # dynamic cache folder path
    cache_folder = os.path.join(folder_path, "cache")

    # create cache folder if it doesn't exist
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)
        print(f"{cache_folder} created")
    else:
        print(f"{cache_folder} already exists")

    # loop through all files in folder
    for filename in os.listdir(folder_path):
        if any(filename.endswith(ext) for ext in extensions):  # only process image files
            image_path = os.path.join(folder_path, filename)
            cache_key = f"{filename}_{target_size}"
            cached_image_path = os.path.join(cache_folder, cache_key)

            # check if image is in cache folder
            if os.path.exists(cached_image_path):
                resized_image = cv2.imread(cached_image_path)

            else:
                # if not, resize image
                image = cv2.imread(image_path)
                resized_image = cv2.resize(image, target_size)

                # save resized image to cache folder
                cv2.imwrite(cached_image_path, resized_image)

            resized_images.append(resized_image)
    print("Completed loading images")

    return np.array(resized_images)

def load_training_set(test_folder_path, test_ground_truth_folder_path):
    """
    Load the training set images and masks

    :param folder_path: path to folder containing images and masks
    :return: numpy array of images and masks
    """

    # Check if the folder_path exists
    if not os.path.exists(test_folder_path):
        raise FileNotFoundError(f"The folder path '{test_folder_path}' does not exist.")

    if not os.path.exists(test_ground_truth_folder_path):
        raise FileNotFoundError(f"The folder path '{test_ground_truth_folder_path}' does not exist.")


    # Get list of image and mask filenames
    image_filenames = [f for f in os.listdir(test_folder_path) if f.endswith('.jpg')]
    mask_filenames = [f for f in os.listdir(test_ground_truth_folder_path) if f.endswith('_segmentation.png')]

    # Create sets of the base names without the extensions or '_segmentation'
    image_basenames = set([os.path.splitext(f)[0] for f in image_filenames])
    mask_basenames = set([os.path.splitext(f)[0].replace('_segmentation', '') for f in mask_filenames])

    # Check if the sets are equal, indicating a one-to-one correspondence
    if image_basenames != mask_basenames:
        raise ValueError("The images and masks are not in a one-to-one correspondence.")

    return load_images(test_folder_path), load_images(test_ground_truth_folder_path)


# # test load_images function
# if __name__ == '__main__':
#     folder_path = "E:/comp3710/ISIC2018/ISIC2018_Task1-2_Validation_Input"
#     target_size = (256, 256)
#     resized_images = load_images(folder_path, target_size)
#     print(resized_images.shape)
#     # output: (2594, 256, 256, 3), 2594 images of size 256x256 with 3 channels

# if __name__ == '__main__':
#     test_folder_path = "E:/comp3710/ISIC2018/ISIC2018_Task1-2_Validation_Input"
#     test_ground_truth_folder_path = "E:/comp3710/ISIC2018/ISIC2018_Task1_Validation_GroundTruth"
#     images, masks = load_training_set(test_folder_path, test_ground_truth_folder_path)
#     print(images.shape)
#     print(masks.shape)
#     # output: (2594, 256, 256, 3), 2594 images of size 256x256 with 3 channels
#     # output: (2594, 256, 256, 3), 2594 images of size 256x256 with 3 channels




