import os
import random
import matplotlib.pyplot as plt
import datetime


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.plot(iteration, loss)
    # plt.savefig(f"./assets/train_loss.png")


def save_plot(iteration, loss):
    with open(f"./assets/iteration_{datetime.datetime.now()}.txt", "w") as output:
        output.write(str(iteration))
    with open(f"./assets/loss_{datetime.datetime.now()}.txt", "w") as output:
        output.write(str(loss))
