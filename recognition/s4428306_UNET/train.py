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

#Using this to pass to accuracy as suggested by:
#https://stackoverflow.com/questions/52123670/neural-network-converges-too-fast-and-predicts-blank-results
def dice_coef(y_true, y_pred):
    numerator = tf.math.reduce_sum(y_true * y_pred)
    denominator = tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred)
    total = numerator / denominator
    return 2 * total

#Going to make this exclusively 1 class now.
class Dice(tf.keras.losses.Loss):
    def __call__(self, y_true, y_pred, sample_weight=None):
        #NOTE: Want to know if whole batches are being passed to this function.
        #      If so, surely it would have complained by now, right?
        #      Turns out the predictions should have been rounded.
        #      Rounding them makes this function non-differentiable.
        #y_pred = tf.math.round(y_pred)
        #print("Y_TRUE SHAPE", y_true.shape)
        #print("Y_PRED SHAPE", y_pred.shape)
        #Compute Dice loss.
        #NOTE: Is the way this for loop is created not differentiable?
        #      May need to look at other dice function implementations.
        #for i in range(self.k):
        #Will these be different shapes if batches are used?
        #y_true_i = y_true[:, :, i]
        #y_pred_i = y_pred[:, :, i]
        #numerator = tf.math.reduce_sum(y_true * y_pred)
        #denominator = tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred)
        #total = numerator / denominator
        #return -2 * total
        return -1 * dice_coef(y_true, y_pred)

def dice_acc(y_true, y_pred):
    return dice_coef(y_true, tf.math.round(y_pred))

#NOTE: Not actually dice related.
#      Just makes sure accuracy is measured properly.
class DiceAccuracy(tf.keras.metrics.Accuracy):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.round(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)

#Plotting code taken from:
#https://stackoverflow.com/questions/41908379/keras-plot-training-validation-and-test-set-accuracy
def create_plots(history):
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
    #Load data.
    train_batches, val_batches, test_batches = preprocessing()
    #Set up training parameters.
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    TRAIN_N = 1796
    TRAIN_LENGTH = 2 * TRAIN_N
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, STEPS_PER_EPOCH, 0.985)
    #NOTE: No l2 decay yet.
    a_opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    dice_loss = Dice()
    #dice_accuracy = DiceAccuracy()
    #Build model.
    iunet_model = build_improved_unet_model()
    iunet_model.summary()
    iunet_model.compile(optimizer=a_opt, #tf.keras.optimizers.Adam(),
                        loss=dice_loss,
                        metrics=[dice_acc]) #"accuracy")
    model_history = iunet_model.fit(train_batches,
                                    epochs=NUM_EPOCHS,
                                    steps_per_epoch=STEPS_PER_EPOCH,
                                    validation_data=val_batches)
    iunet_model.save("/home/Student/s4428306/report/iunet_model.keras", save_format="keras")
    create_plots(model_history)

if __name__ == "__main__":
    train_main()

