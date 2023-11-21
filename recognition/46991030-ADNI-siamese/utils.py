"""
utils.py: Utility functions

This file contains utility functions used throughout the project.
"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_training_history(history: tf.keras.callbacks.History, model: str):
    """
    Plots the given training/validation accuracy and loss over the training epochs from the given History object.

    Args:
        history (tf.keras.callbacks.History): The History object to plot.
    """
    plt.figure(figsize=(12, 10))
    plt.title("Loss over training epochs")
    plt.plot(history.history["loss"], lw=2, label="Training loss")
    plt.plot(history.history["val_loss"], lw=2, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"graphs/{model}-loss.png")

    plt.figure(figsize=(12, 10))
    plt.title("Accuracy over training epochs")
    plt.plot(
        100 * np.array(history.history["accuracy"]), lw=2, label="Training accuracy"
    )
    plt.plot(
        100 * np.array(history.history["val_accuracy"]),
        lw=2,
        label="Validation accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="lower right")
    plt.savefig(f"graphs/{model}-accuracy.png")
