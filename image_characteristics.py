from PIL import Image
import os
import matplotlib.pyplot as plt

image_directory = '/Users/noammendelson/Documents/REPORT-COMP3710/AD_NC/train/AD'
image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]

widths = []
heights = []

for image_file in image_files:
    with Image.open(os.path.join(image_directory, image_file)) as img:
        width, height = img.size
        widths.append(width)
        heights.append(height)

# Now compute and print statistics
print(f"Mean Width: {sum(widths) / len(widths)}")
print(f"Mean Height: {sum(heights) / len(heights)}")
print(f"Max Width: {max(widths)}")
print(f"Max Height: {max(heights)}")
print(f"Min Width: {min(widths)}")
print(f"Min Height: {min(heights)}")

# Plotting histogram
plt.hist(widths, bins=20, alpha=0.5, label='Width')
plt.hist(heights, bins=20, alpha=0.5, label='Height')
plt.legend(loc='upper right')
plt.title("Distribution of Image Dimensions")
plt.xlabel("Pixels")
plt.ylabel("Number of Images")
plt.show()
