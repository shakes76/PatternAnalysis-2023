import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def eval():
    '''
    Only evaluate after all the training, or it will not work.
    An animation of all the recorded generated pictures will be shown.
    :return: None
    '''

    if not os.path.exists("./Pic"):
        raise NotImplementedError
    else:
        image_dir = "./Pic"

        image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

        image_files.sort()

        # Create a figure and axis for the animation
        fig, ax = plt.subplots()
        ax.set_axis_off()
        fig.tight_layout()

        # Function to update the animation
        def update(frame):
            img = Image.open(os.path.join(image_dir, image_files[frame]))
            ax.clear()
            ax.imshow(img)

        # Create the animation
        animation = FuncAnimation(fig, update, frames=len(image_files), repeat=False)
        plt.show()


if __name__ == "__main__":
    eval()