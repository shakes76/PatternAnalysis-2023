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


# test function
if __name__ == '__main__':
    folder_path = "E:/comp3710/ISIC2018/ISIC2018_Task1-2_Test_GroundTruth"
    target_size = (256, 256)
    resized_images = load_images(folder_path, target_size)
    print(resized_images.shape)
    # output: (2594, 256, 256, 3)
