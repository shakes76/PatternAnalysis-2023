""" Utility functions for model training and evaluation. """

import matplotlib.pyplot as plt

def plot_data(train_data, 
                    test_data, 
                    heading,
                    labels=None):
    if labels is None:
        labels = ['Train', 'Test']

    # Check if the lengths of input lists match
    if len(train_data) != len(test_data):
        raise ValueError("Train and test accuracy lists must have the same length.")

    # Create x-axis values (epochs)
    epochs = range(1, len(train_data) + 1, 1)

    # Plot train and test accuracies
    plt.plot(epochs, train_data, label=labels[0])
    plt.plot(epochs, test_data, label=labels[1])

    # Set plot labels and legend
    plt.xlabel('Epoch')
    plt.ylabel(f'{heading}')
    plt.legend(loc='best')

    # Show the plot
    plt.grid(True)
    plt.title(f'{labels[0]} and {labels[1]} {heading} Over Epochs')
    plt.show()

# Function uses a modification of the code found in 
# https://www.learnpytorch.io/08_pytorch_paper_replicating/#44-flattening-the-patch-embedding-with-torchnnflatten
def show_patched_image(img, img_size, patch_size):
    """ Plot a permuted image as a series of patches. """
    # Setup hyperparameters and make sure img_size and patch_size are compatible
    num_patches = img_size/patch_size
    
    assert img_size % patch_size == 0, "Image size must be divisible by patch size"

    # Create a series of subplots
    fig, axs = plt.subplots(nrows=img_size // patch_size, # need int not float
                            ncols=img_size // patch_size, 
                            figsize=(num_patches, num_patches),
                            sharex=True,
                            sharey=True)
    
    # Set the spacing between subplots
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # Loop through height and width of image
    for i, patch_height in enumerate(range(0, img_size, patch_size)): # iterate through height
        for j, patch_width in enumerate(range(0, img_size, patch_size)): # iterate through width
            
            # Plot the image patch
            axs[i, j].imshow(img[patch_height:patch_height+patch_size,  # iterate through height 
                                            patch_width:patch_width+patch_size], # iterate through width
                                            cmap = "gray") 
            
            # Set up label information, remove the ticks for clarity and set labels to outside
            axs[i, j].set_ylabel(i+1, 
                                rotation="horizontal", 
                                horizontalalignment="right", 
                                verticalalignment="center") 
            axs[i, j].set_xlabel(j+1) 
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            axs[i, j].label_outer()

    # Set a super title
    fig.suptitle("Example patched image", fontsize=16)
    plt.show()