from dataset import *
from modules import *
import tensorflow as tf
from tensorflow import keras
from tensorflow import math
from tensorflow.keras import layers
from matplotlib import pyplot as plt

#TODO: Clean up this file.
#      Reference UNet model this is based on.
#      Write documentation for these functions/classes.

class Dice(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name=None, k=1):
        super().__init__(reduction=reduction, name=name)
        self.k = k

    def call(self, y_true, y_pred):
        #NOTE: Want to know if whole batches are being passed to this function.
        #      If so, surely it would have complained by now, right?
        print("Y_TRUE SHAPE", y_true.shape)
        print("Y_PRED SHAPE", y_pred.shape)
        #Compute Dice loss.
        total = 0
        for i in range(self.k):
            #Will these be different shapes if batches are used?
            y_true_i = y_true[:, :, i]
            y_pred_i = y_pred[:, :, i]
            numerator = tf.math.reduce_sum(y_true_i * y_pred_i)
            denominator = tf.math.reduce_sum(y_true_i) + tf.math.reduce_sum(y_pred_i)
            total += numerator / denominator
        return -1 * (2 / self.k) * total

    def get_config(self):
        config = super().get_config()
        config.update({"k": self.k})
        return config

#Plotting code taken from:
#https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
def create_plots(history):
    #Save accuracy plot.
    plt.figure(0)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
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
    #Load data.
    train_batches, val_batches, test_batches = preprocessing()
    #Set up training parameters.
    BATCH_SIZE = 64
    NUM_EPOCHS = 10
    TRAIN_N = 1796
    TRAIN_LENGTH = 2 * TRAIN_N
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, STEPS_PER_EPOCH, 0.985)
    #NOTE: No l2 decay yet.
    #a_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    dice_loss = Dice(k=2)
    #Build model.
    iunet_model = build_improved_unet_model()
    iunet_model.summary()
    iunet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss=dice_loss,
                        metrics="accuracy")
    model_history = iunet_model.fit(train_batches,
                                    epochs=NUM_EPOCHS,
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    validation_data=val_batches)
    iunet_model.save("/home/Student/s4428306/report/iunet_model.keras", save_format="keras")
    create_plots(model_history)

if __name__ == "__main__":
    train_main()

