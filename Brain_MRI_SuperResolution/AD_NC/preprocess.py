import os
from PIL import Image


def downsample_images(input_folder, factor=4):
    print(f"Processing images in folder: {input_folder}")

    # Counters for the total number of files and the number of files processed
    total_files = 0
    processed_files = 0

    for root, dirs, files in os.walk(input_folder):
        total_files += len(files)
        for file in files:
            if file.endswith('.png'):
                img_path = os.path.join(root, file)
                print(f"Processing image: {img_path}")

                img = Image.open(img_path)

                # Downsample the image
                img_resized = img.resize((img.width // factor, img.height // factor), Image.ANTIALIAS)
                img_resized.save(img_path)

                processed_files += 1
                print(f"Processed {processed_files} out of {total_files}")

    print(f"Completed processing images in folder: {input_folder}")


if __name__ == "__main__":
    downsample_images("data/ad_nc/train")
    downsample_images("data/ad_nc/test")
