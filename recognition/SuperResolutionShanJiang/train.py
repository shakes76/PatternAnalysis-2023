from utils import *
from modules import *
from dataset import *

from tensorflow import keras
from keras import layers
from keras.utils import load_img
import os
import math

upscale_factor = 4
# define test data from test AD 
test_path = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/original/test/AD'
test_img_paths = sorted(
    [
        os.path.join(test_path, fname)
        for fname in os.listdir(test_path)
        if fname.endswith(".jpeg")
    ]
)
class ESPCNCallback(keras.callbacks.Callback):
    """
    Custom Keras callback for monitoring and displaying PSNR during training.
    """
    def __init__(self):
        super().__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

    # Initialise a array to store epoch PSNR value when each epoch begins
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []
    
    # Print Mean PSNR for when each epoch ends
    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        # if epoch % 20 == 0:
        #     prediction = upscale_image(self.model, self.test_img)
            # plot_results(prediction, "epoch-" + str(epoch), "prediction")
    
    # Store PSNR value when each test epoch ends
    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

# Stop training when loss does not improve for 10 consecutive epochs         
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

# Define path to save model parameters
checkpoint_filepath = "D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/tmp/checkpoint"

# Save model parameters at checkpoint during training
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

# Initialise a model
model = get_model(upscale_factor=upscale_factor, channels=1)
model.summary()
callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

#Train the model
epochs = 100

model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
)

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)

        

