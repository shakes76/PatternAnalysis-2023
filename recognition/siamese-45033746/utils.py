import os
import random
import matplotlib.pyplot as plt
import datetime


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()
    # plt.savefig(f"./assets/train_loss.png")

def save_plot(iteration: [], loss: [], name: str):
    with open(f"./assets/{name}_iteration_{datetime.datetime.now()}.txt", "w") as output:
        output.write(str(iteration))
    with open(f"./assets/{name}_loss_{datetime.datetime.now()}.txt", "w") as output:
        output.write(str(loss))


if __name__ == '__main__':
    data = open("./assets/test_x.txt", "r")
    info = data.read()
    counter_list = info.replace('\n', '').split(",")
    counter_list = list(map(int, counter_list))
    data.close()

    data = open("./assets/test_y.txt", "r")
    info = data.read()
    train_loss_list = info.replace('\n', '').split(",")
    train_loss_list = list(map(float, train_loss_list))
    data.close()

    data = open("./assets/val_x.txt", "r")
    info = data.read()
    val_counter_list = info.replace('\n', '').split(",")
    val_counter_list = list(map(int, val_counter_list))
    data.close()

    data = open("./assets/val_y.txt", "r")
    info = data.read()
    val_loss_list = info.replace('\n', '').split(",")
    val_loss_list = list(map(float, val_loss_list))
    data.close()

    show_plot(counter_list, train_loss_list)
    show_plot(val_counter_list, val_loss_list)
