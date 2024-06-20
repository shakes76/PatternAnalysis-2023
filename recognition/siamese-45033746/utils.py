import os
import random
import matplotlib.pyplot as plt
import datetime

"""
utils.py

utilities folder for helper functions
"""


def show_plot(iteration, loss):
    """
    show loss plot
    :param iteration: number of batches
    :param loss: loss for each batch
    """
    plt.plot(iteration, loss)
    plt.show()


def save_plot(iteration: [], loss: [], name: str):
    """
    Saves loss plot to ./assets
    :param iteration: number of batches
    :param loss: loss for each batch
    :param name: Name of batch set being tested
    """
    plt.plot(iteration, loss)
    plt.savefig(f"./assets/{name}.png")
