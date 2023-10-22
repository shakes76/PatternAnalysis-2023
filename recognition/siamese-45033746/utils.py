import os
import random
import matplotlib.pyplot as plt

def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.plot(iteration, loss)
    # plt.savefig(f"./assets/train_loss.png")
