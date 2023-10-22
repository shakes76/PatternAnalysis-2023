from utils import *
from modules import *
from dataset import *

from tensorflow import keras
from keras import layers
from keras.utils import load_img
import os
import math
import matplotlib.pyplot as plt

upscale_factor = 4
loss_plot_path = 'D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/loss_plot/'

train_loss_history = []
valid_loss_history = []
train_psnr_history = []

# Custom Keras callback for monitoring and displaying PSNR during training.
class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    # Initialise an array to store epoch PSNR value when each epoch begins
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []
    
    # Print mean training PSNR for when each epoch ends
    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        train_loss_history.append(logs['loss'])
        valid_loss_history.append(logs['val_loss'])
        train_psnr_history.append(np.mean(self.psnr))

        
        if epoch % 9 == 0 and epoch!= 0:
            # Plot loss history after every 10 epoch and save the plot
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_history, label='Training Loss', color='blue')
            plt.plot(valid_loss_history, label='Validation Loss', color='red')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(loss_plot_path + 'epoch' + str(epoch+1) + '.png')

    # Store training PSNR value when each test epoch ends
    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

# Stop training when loss does not improve for 10 consecutive epochs         
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

# Path to save model parameters
checkpoint_filepath = "D:/temporary_workspace/comp3710_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/checkpoint/"
# checkpoint_filepath = "H:/final_project/PatternAnalysis_2023_Shan_Jiang/recognition/SuperResolutionShanJiang/tmp/checkpoint/"

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
# model.summary()

# Define callbacks, loos function and optimiser
callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback] 
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)

#Train and validate the model
epochs = 60

model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
)


# Initialise a model
model = get_model(upscale_factor=upscale_factor, channels=1)

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)

# Testing metrics
total_bicubic_psnr = 0.0 # PSNR of downsampled image
total_test_psnr = 0.0 # PSNR of model output

# Dowansample resolution of iamges by factor of 4, then predict higher resolution image using the model
for index, test_img_path in enumerate(get_test_img_paths()):
    img = load_img(test_img_path)
    lowres_input = get_lowres_image(img, upscale_factor) # downsample
    w = lowres_input.size[0] * upscale_factor
    h = lowres_input.size[1] * upscale_factor
    highres_img = img.resize((w, h))
    prediction = upscale_image(model, lowres_input) # Predict
    lowres_img = lowres_input.resize((w, h))
    lowres_img_arr = img_to_array(lowres_img)
    highres_img_arr = img_to_array(highres_img)
    predict_img_arr = img_to_array(prediction)
    bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
    test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

    total_bicubic_psnr += bicubic_psnr
    total_test_psnr += test_psnr

print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))

        

