#Extensive use of tensorflow and keras documentation was used to write this code.
#Code in this section is based on code from:
#https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/

from dataset import *
from modules import *
import tensorflow as tf
from tensorflow import keras
from tensorflow import math
from tensorflow.keras import layers
from matplotlib import pyplot as plt

#Using this to pass to accuracy as suggested by:
#https://stackoverflow.com/questions/52123670/neural-network-converges-too-fast-and-predicts-blank-results
def dice_coef(y_true, y_pred):
    """
    Calculates the dice coefficient between the true mask and models prediction.
    """
    numerator = tf.math.reduce_sum(y_true * y_pred)
    denominator = tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred)
    total = numerator / denominator
    return 2 * total

class Dice(tf.keras.losses.Loss):
    """
    A class that encapsulates the dice coefficient as a loss function.
    """
    def __call__(self, y_true, y_pred, sample_weight=None):
        return -1 * dice_coef(y_true, y_pred)

def dice_acc(y_true, y_pred):
    """
    Accuracy function for model compilation.
    """
    return dice_coef(y_true, tf.math.round(y_pred))

#Plotting code taken from:
#https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
def create_plots(history):
    """
    Saves plots of the model's training history for loss and accuracy.
    """
    #Save accuracy plot.
    plt.figure(0)
    plt.plot(history.history['dice_acc'])
    plt.plot(history.history['val_dice_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("/home/Student/s4428306/report/accuracy_plot.png")
    #Save loss plot.
    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("/home/Student/s4428306/report/loss_plot.png")

def train_main():
    """
    Trains the model, saves it, and saves the training graphs.
    """
    #Load data.
    train_batches, val_batches, test_batches = preprocessing()
    #Set up training parameters.
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    TRAIN_N = 1796
    TRAIN_LENGTH = 2 * TRAIN_N
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, STEPS_PER_EPOCH, 0.985)
    a_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    dice_loss = Dice()
    #Build model.
    iunet_model = build_improved_unet_model()
    iunet_model.summary()
    iunet_model.compile(optimizer=a_opt,
                        loss=dice_loss,
                        metrics=[dice_acc])
    #Train and save.
    model_history = iunet_model.fit(train_batches,
                                    epochs=NUM_EPOCHS,
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    validation_data=val_batches)
    iunet_model.save("/home/Student/s4428306/report/iunet_model.keras", save_format="keras")
    #Save graphs.
    create_plots(model_history)

if __name__ == "__main__":
    train_main()

