import matplotlib.pyplot as plt
import numpy as np

def plot_accuracy(train_acc, val_acc, save_path):
    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(1, len(val_acc) + 1), val_acc, label="Validation Accuracy")
    plt.plot(np.arange(1, len(train_acc) + 1), train_acc, label="Training Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accurarcy")
    plt.title("Training and Validation Accuracy across Epochs")
    plt.legend(loc="lower right", fontsize="x-large")
    plt.savefig(save_path)
    plt.show()


def plot_loss(train_loss, val_loss, save_path):
    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label="Training Loss")
    plt.plot(np.arange(1, len(val_loss) + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss across Epochs")
    plt.legend(loc="upper right", fontsize="x-large")
    plt.savefig(save_path)
    plt.show()