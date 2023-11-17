from matplotlib import pyplot as plt


def main():

    epoch_losses = list()
    g_losses = list()
    d_losses = list()

    epoch_lrs = list()
    g_lrs = list()
    d_lrs = list()

    epoch_lim = 100
    count = 1

    f = open("out_gan.txt")
    lines = f.readlines()

    for i, line in enumerate(lines):

        if "epoch" in line:
            
            epoch = int(line.split(",")[0].split(":")[1].split("/")[0].strip())

            if epoch > epoch_lim:
                break

            step = int(line.split(",")[1].split(":")[1].split("/")[0].strip())
            steps = int(line.split(",")[1].split(":")[1].split("/")[1].strip())
            epoch += step/steps
            epoch_losses.append(epoch - 1)

            d_losses.append(float(line.split(",")[2].split(":")[1].strip()))
            g_losses.append(float(line.split(",")[3].split(":")[1].strip()))

        if "lr" in line:

            if count > epoch_lim:
                break

            epoch_lrs.append(count)

            d_lrs.append(float(line.split(",")[0].split(":")[1].strip()))
            g_lrs.append(float(line.split(",")[1].split(":")[1].strip()))

            count += 1

    plt.style.use("dark_background")

    plt.plot(epoch_losses, d_losses, color="red")
    plt.plot(epoch_losses, g_losses, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(["Discriminator Loss", "Generator Loss"])
    plt.title("Training Loss Plot")
    plt.show()

    plt.plot(epoch_lrs, d_lrs, color="red")
    plt.plot(epoch_lrs, g_lrs, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.legend(["Discriminator LR", "Generator LR"])
    plt.title("Variable Learning Rate Plot")
    plt.show()


if __name__ == "__main__":
    main()