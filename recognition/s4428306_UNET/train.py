from dataset import *
from modules import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#TODO: Clean up this file.

#!!!
#Going to try and train it using the built in functions.
#!!!
#Load data.
train_batches, val_batches, test_batches = preprocessing()

#Build model.
iunet_model = build_improved_unet_model()
iunet_model.summary()
iunet_model.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss="sparse_categorical_crossentropy",
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

print(model_history)
print("SUCCESS")

