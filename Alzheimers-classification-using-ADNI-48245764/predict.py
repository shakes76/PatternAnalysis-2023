import train
import modules
import matplotlib.pyplot as plt
import os

def main():
    model = modules.transformer()
    performance = train.run_model(model)

    # Plot accuracy
    plt.plot(performance.performance['accuracy'])
    plt.plot(performance.performance['val_accuracy'])
    plt.title("Accuracy Plot")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(['Training', 'Validation'], loc='upper left')
    accuracy_plot_path = os.path.join(os.path.dirname(__file__), "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    plt.show()

    # Plot loss
    plt.plot(performance.performance['loss']) 
    plt.plot(performance.performance['val_loss'])
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(['train', 'validation'], loc='upper left')
    # Save loss plot as an image
    loss_plot_path = os.path.join(os.path.dirname(__file__), "loss_plot.png")
    plt.savefig(loss_plot_path)
    plt.show()

if __name__ == "__main__":
    main()
