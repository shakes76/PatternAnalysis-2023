from dataset import *
from modules import *
import tensorflow as tf
from tensorflow import keras
from tensorflow import math
from tensorflow.keras import layers

#TODO: Clean up this file.
#      Reference UNet model this is based on.
#      Write documentation for Dice loss function.

class Dice(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        #Number of classes.
        #NOTE: Number of classes could be a class parameter passed to the init function.
        k = y_true.shape[-1]
        #Compute Dice loss.
        total = 0
        for i in range(k):
            #Will these be different shapes if batches are used?
            y_true = y_true[:, :, i]
            y_pred = y_pred[:, :, i]
            numerator = tf.math.reduce_sum(y_true * y_pred)
            denominator = tf.math.reduce_sum(y_true) + tf.math.reduce_sum(y_pred)
            total += numerator / denominator
        #NOTE: Unsure about whether this should be negative or not.
        return -1 * (2 / k) * total

dice_loss = Dice()

#!!!
#Going to try and train it using the built in functions.
#!!!
#Load data.
train_batches, val_batches, test_batches = preprocessing()

#Build model.
iunet_model = build_improved_unet_model()
iunet_model.summary()
iunet_model.compile(optimizer=tf.keras.optimizers.Adam(),
#                    loss="sparse_categorical_crossentropy",
                    loss=dice_loss,
                    metrics="accuracy")

#Train model.
BATCH_SIZE = 64
#NOTE: Going to make this brief for testing purposes.
NUM_EPOCHS = 2
TRAIN_N = 1796
TRAIN_LENGTH = 2 * TRAIN_N
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
model_history = iunet_model.fit(train_batches,
                                epochs=NUM_EPOCHS,
                                steps_per_epoch=STEPS_PER_EPOCH,
                                validation_data=val_batches)
iunet_model.save("/home/Student/s4428306/report/iunet_model_attempt1.keras", save_format="keras")

print(model_history)
print("SUCCESS")

