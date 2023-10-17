import matplotlib as plt
import torch


def display_images(images, labels, predictions):
    plt.figure(figsize=(10, 7))
    for i, image in enumerate(images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            image[0], cmap="gray"
        )  # Assuming grayscale images. If RGB, remove [0].
        plt.title(f"True: {labels[i]}, Predicted: {predictions[i]}")
        plt.axis("off")
    plt.show()


def display_feature_maps(model, image, layers_to_display):
    plt.figure(figsize=(20, 20))
    with torch.no_grad():
        for layer in layers_to_display:
            x = layer(image)
            # Assuming the feature maps are in shape (batch, channels, height, width)
            for i in range(x.size(1)):
                plt.subplot(8, 8, i + 1)  # assuming you have 64 filters
                plt.imshow(x[0][i].cpu().numpy(), cmap="gray")
                plt.axis("off")
    plt.show()


def backward_hook(module, grad_input, grad_output):
    print("Inside " + module.__class__.__name__ + " backward hook:")
    print("grad_input", grad_input)
    print("grad_output", grad_output)
    print("=" * 50)
    return grad_input


# We use this function to generate a normalization value for the dataset
def compute_mean_std(loader):
    """
    Compute the mean and standard deviation of the dataset.
    """
    mean = 0.0
    squared_mean = 0.0
    std = 0.0
    for images, _ in loader:
        mean += images.mean()
        squared_mean += (images**2).mean()

    mean /= len(loader)
    squared_mean /= len(loader)
    std = (squared_mean - mean**2) ** 0.5
    return mean.item(), std.item()
