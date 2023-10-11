"""
This code is for testing dataset.py
"""

# Get the train and test loaders
train_loader, validation_loader = get_train_and_validation_loaders()
test_loader = get_test_loader()

# Extract a single batch from the train loader
data_iter = iter(train_loader)
downscaled_images, original_images = next(data_iter)

print(original_images[0].shape)

# Convert the images from tensor format back to PIL for display
def tensor_to_PIL(tensor):
    tensor = (tensor + 1) / 2.0
    tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor)

# Plot the images (plot 5 to ensure it is working)
num_images_to_display = 5

for i in range(num_images_to_display):
    plt.figure(figsize=(10,5))
    
    # Display downscaled image
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_PIL(downscaled_images[i]))
    plt.title("Downscaled Image")
    
    # Display original image
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_PIL(original_images[i]))
    plt.title("Original Image")
    
    plt.show()

"""
End code
"""