import os
from PIL import Image

def downsample_images(input_folder, output_folder, factor=4):
    print(f"Processing images in folder: {input_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_files = 0
    processed_files = 0

    for root, dirs, files in os.walk(input_folder):
        total_files += len(files)
        for file in files:
            if file.endswith('.jpeg'):
                img_path = os.path.join(root, file)
                print(f"Processing image: {img_path}")

                img = Image.open(img_path)
                img_resized = img.resize((img.width // factor, img.height // factor), Image.ANTIALIAS)

                # Save to the new directory
                save_path = os.path.join(output_folder, file)
                img_resized.save(save_path)

                processed_files += 1
                print(f"Processed {processed_files} out of {total_files}")

    print(f"Completed processing images in folder: {input_folder}")


if __name__ == "__main__":
    downsample_images("train", "train_downsampled")
    downsample_images("test", "test_downsampled")
