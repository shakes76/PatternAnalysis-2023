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