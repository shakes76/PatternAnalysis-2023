import matplotlib.pyplot as plt

"""
Generate training plots
"""

def process_output(text_file):
    """
    Process the output text file from training and return the epochs, 
    training loss, training accuracy, validation loss, and validation 
    accuracy as lists.
    
    Args:
        text_file (str): The path to the output text file.
    
    Returns:
        epochs (list): The list of epochs.
        train_loss (list): The list of training loss values.
        train_accuracy (list): The list of training accuracy values.
        valid_loss (list): The list of validation loss values.
        valid_accuracy (list): The list of validation accuracy values.
    """
    epochs = []
    train_loss = []
    train_accuracy = []
    valid_loss = []
    valid_accuracy = []

    # Open and read the text file
    with open(text_file, "r") as file:
        lines = file.readlines()

    current_epoch = None
    for line in lines:
        if line.startswith("INFO:root:Epoch: "):
            current_epoch = int(line.split()[-1])
            epochs.append(current_epoch)
        elif line.startswith("Train loss: "):
            train_loss.append(float(line.split()[-1]))
        elif line.startswith("Train Accuracy: "):
            train_accuracy.append(float(line.split()[-1]))
        elif line.startswith("Valid Loss: "):
            valid_loss.append(float(line.split()[-1]))
        elif line.startswith("Valid Accuracy: "):
            valid_accuracy.append(float(line.split()[-1]))

    return epochs, train_loss, train_accuracy, valid_loss, valid_accuracy

def plot(text_file, saved_file):
    """
    Plot the training and validation loss and accuracy from the output text file.
    """
    epochs, train_loss, train_accuracy, valid_loss, valid_accuracy = process_output(text_file)
    # Plot training loss
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Training Loss", marker='o', linestyle='-')
    plt.plot(epochs, valid_loss, label="Validation Loss", marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label="Training Accuracy", marker='o', linestyle='-')
    plt.plot(epochs, valid_accuracy, label="Validation Accuracy", marker='o', linestyle='-')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(saved_file)

if __name__ == "__main__":
    plot(text_file="out/4.txt", saved_file="plots/plot4.png")