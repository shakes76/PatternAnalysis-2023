from dataset import *
from utils import *
from modules import *
import matplotlib.pyplot as plt

loss_list = []

class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.test_img = get_lowres_image(load_img(test_img_paths[0]), upscale_factor)

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        loss_list.append(logs["loss"])
        if epoch % 20 == 0:
            prediction = upscale_image(self.model, self.test_img)
            #plot_results(prediction, "epoch-" + str(epoch), "prediction")

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))
        
        
early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)

checkpoint_filepath = Path(os.path.join(current_directory,"/saved_model/checkpoint/"))

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="loss",
    mode="min",
    save_best_only=True,
)

model = get_model(upscale_factor=upscale_factor, channels=1)
model.summary()

callbacks = [ESPCNCallback(), early_stopping_callback, model_checkpoint_callback]
loss_fn = keras.losses.MeanSquaredError()
optimizer = keras.optimizers.Adam(learning_rate=0.001)


# training
epochs = 25

model.compile(
    optimizer=optimizer, loss=loss_fn,
)

model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=valid_ds, verbose=2
)

print("Training", loss_list)



# Create the plot
plt.plot(loss_list)

# Add labels and title (optional)
plt.xlabel('epoch')
plt.ylabel('loss value')
plt.title('loss value against training')

# Show the plot
plt.show()
plt.savefig('line_plot.png', dpi=300)

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)


total_bicubic_psnr = 0.0
total_test_psnr = 0.0

# testing
# for index, test_img_path in enumerate(test_img_paths):
#     img = load_img(test_img_path)
#     lowres_input = get_lowres_image(img, upscale_factor)
#     w = lowres_input.size[0] * upscale_factor
#     h = lowres_input.size[1] * upscale_factor
#     highres_img = img.resize((w, h))
#     prediction = upscale_image(model, lowres_input)
#     lowres_img = lowres_input.resize((w, h))
    
#     lowres_img.save(f"recognition/Normal_Difficulty/test_result/{index}_lowres_image.jpeg")  # Change the file extension to match the image format
#     highres_img.save(f"recognition/Normal_Difficulty/test_result/{index}_highres_image.jpeg")
#     prediction.save(f"recognition/Normal_Difficulty/test_result/{index}_predicted_image.jpeg")
        
#     lowres_img_arr = img_to_array(lowres_img)
#     highres_img_arr = img_to_array(highres_img)
#     predict_img_arr = img_to_array(prediction)
    
    
#     bicubic_psnr = tf.image.psnr(lowres_img_arr, highres_img_arr, max_val=255)
#     test_psnr = tf.image.psnr(predict_img_arr, highres_img_arr, max_val=255)

#     total_bicubic_psnr += bicubic_psnr
#     total_test_psnr += test_psnr


# print("Avg. PSNR of lowres images is %.4f" % (total_bicubic_psnr / 10))
# print("Avg. PSNR of reconstructions is %.4f" % (total_test_psnr / 10))