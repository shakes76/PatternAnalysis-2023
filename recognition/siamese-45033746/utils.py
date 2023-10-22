import os
import random
import matplotlib.pyplot as plt
import datetime


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()

def save_plot(iteration: [], loss: [], name: str):
    plt.plot(iteration, loss)
    plt.savefig(f"./assets/{name}.png")
