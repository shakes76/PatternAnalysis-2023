import train
import modules
import matplotlib.pyplot as plt
import os

def main():
    """
    Main function to train a model, plot accuracy and loss, and save the plots as images.

    This function loads a neural network model using modules.transformer(),
    trains the model using train.run_model(), and plots the accuracy and loss
    during training. The resulting plots are saved as images in the same directory.

    Note:
        - The 'accuracy_plot.png' image will contain the accuracy plot.
        - The 'loss_plot.png' image will contain the loss plot.
    """

    model = modules.transformer()
    performance = train.run_model(model)

    # Plot Accuracy
    plt.plot(performance.performance['accuracy'])
    plt.plot(performance.performance['val_accuracy'])
    plt.title("Accuracy Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    accuracy_plot_path = os.path.join(os.path.dirname(__file__), "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    plt.show()

    # Plot Loss
    plt.plot(performance.performance['loss']) 
    plt.plot(performance.performance['val_loss'])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_plot_path = os.path.join(os.path.dirname(__file__), "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.show()

if __name__ == "__main__":
    main()
