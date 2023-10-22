import os
import random
import matplotlib.pyplot as plt
import datetime


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.plot(iteration, loss)
    # plt.savefig(f"./assets/train_loss.png")

def save_plot(iteration: [], loss: [], name: str):
    with open(f"./assets/{name}_iteration_{datetime.datetime.now()}.txt", "w") as output:
        output.write(str(iteration))
    with open(f"./assets/{name}_loss_{datetime.datetime.now()}.txt", "w") as output:
        output.write(str(loss))
