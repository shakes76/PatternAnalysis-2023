import tensorflow as tf
import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from modules import improved_unet
from dataset import process_images, create_ds, img_height, img_width
from train import dice_sim_coef, initialise_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

"""Graph Constants"""
# Number of examples to display
num_display_examples = 3
# If a subset of the original images and masks is to be graphed and printed
print_original_images = True
# If a subset of the processed images and masks is to be graphed and printed
print_processed_images = True
# If a subset of the predicted images and masks is to be graphed and printed
print_predicted_images = True
# If the graphs should be saved as .png files in the Results subdirectory
save_photos = True

"""Model Saving Constants"""
# If a pre-trained model should be used
use_saved_model = False
# If the model being trained should be saved (Note: Only works if use_saved_model = True)
save_model = True

"""Model Training Constants"""
# If the images should be shuffled (Note: Masks and their related image are not changed)
shuffle = True
# The dataset split percentage for the training dataset
training_split = 0.8
# The dataset split percentage for the validation dataset
# (Note: The testing dataset will be the remaining dataset once the training and validation datasets have been taken)
validation_split = 0.1
# The shuffle size to be used
shuffle_size = 50

# The height and width of the processed image
img_height = img_width = 256
# The batch size to be used
batch_size = 16
# The number of training epochs
epochs = 10
# The number of times a similar validation dice coefficient score is achieved before training is stopped early
patience = 5


def plot_images(pic_array, rows, index, original, cols=2):
    title = ['Original Input', 'True Mask', 'Predicted Mask']
    for i in range(len(pic_array)):
        plt.subplot(rows, cols, index + 1)
        if index < cols:
            plt.title(title[index])
        if original:
            plt.imshow(mpimg.imread(pic_array[i]))
        else:
            plt.imshow(tf.keras.utils.array_to_img(pic_array[i]))
        plt.axis('off')
        index += 1


def show_original_images(rows, index=0):
    fig = plt.figure(figsize=(5, 5))
    for i in range(rows):
        image, mask = image_file_list[i], mask_file_list[i]
        plot_images([image, mask], rows, index, True)
        index += 2
    plt.show()
    if save_photos:
        fig.savefig('Results/OriginalExample.png', dpi=fig.dpi)


def show_processed_images(rows, ds, index=0):
    fig = plt.figure(figsize=(5, 5))
    for images, masks in ds.take(rows):
        image = images
        mask = masks
        plot_images([image, mask], rows, index, False)
        index += 2
    plt.show()
    if save_photos:
        fig.savefig('Results/ProcessedExample.png', dpi=fig.dpi)


def show_predicted_images(rows, unet_model, index=0):
    fig = plt.figure(figsize=(5, 5))
    for image, mask in test_ds.batch(batch_size).take(num_display_examples):
        pred_mask = tf.cast(unet_model.predict(image), tf.float32)
        plot_images([image[0], mask[0], pred_mask[0]], rows, index, False, cols=3)
        index += 3
    plt.show()
    if save_photos:
        fig.savefig('Results/PredictedExample.png', dpi=fig.dpi)

def plot_performance_loss_model(model_history):
    fig = plt.figure()
    val_loss = model_history.history['val_loss']
    train_loss = model_history.history['loss']
    plt.plot(model_history.epoch, train_loss, 'r', label='Training loss')
    plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.ylim([0, 1])
    plt.legend()
    plt.show()
    if save_photos:
        fig.savefig('Results/ModelLossPerformance.png', dpi=fig.dpi)


def plot_performance_model(model_history):
    fig = plt.figure()
    dice = model_history.history['dice_sim_coef']
    val_dice = model_history.history['val_dice_sim_coef']
    plt.plot(model_history.epoch, dice, 'r', label='Training')
    plt.plot(model_history.epoch, val_dice, 'b', label='Validation')
    plt.title('Dice Similarity Coefficient over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (Dice Similarity Coefficient)')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()
    if save_photos:
        fig.savefig('Results/ModelPerformance.png', dpi=fig.dpi)

# Segments folders into arrays
image_file_list = list(glob.glob('ISIC2018_Task1-2_Training_Input/*.jpg'))
mask_file_list = list(glob.glob('ISIC2018_Task1_Training_GroundTruth/*.png'))


# Create sub directories needed by later processes
try:
    os.mkdir("Results")
except OSError:
    print("Results sub directory already present or could not create folder")

# Show original images and masks
print("Size of Training Pictures: %d\nSize of Segmented Pictures: %d\n"
      % (len(list(image_file_list)), len(list(mask_file_list))))
# Prints a subset of the original images and masks if specified
if print_original_images:
    show_original_images(num_display_examples)
# Creates a dataset that contains all the files
#data_dir = os.getcwd() + '/datasets'
files_ds = tf.data.Dataset.from_tensor_slices((image_file_list, mask_file_list))
files_ds = files_ds.map(lambda x, y: (process_images(x, False), process_images(y, True)),
                        num_parallel_calls=tf.data.AUTOTUNE)

# Prints a subset of the processed images and masks if specified
if print_processed_images:
    show_processed_images(num_display_examples, files_ds)

# Shuffles the dataset if specified
if shuffle:
    files_ds = files_ds.shuffle(shuffle_size)

# Creates datasets of Training, Validation, and Testing data
train_ds, val_ds, test_ds = create_ds()


# Initialise the model
model = initialise_model()
    
# plot model 
# Create the UNet model
unet_model = improved_unet(img_height, img_width, 3)

# Compile the UNet model and add 'dice_sim_coef' as a monitored metric
unet_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_sim_coef])

# Create an EarlyStopping callback to monitor the custom metric
callback = EarlyStopping(monitor='val_dice_sim_coef', patience=patience, mode='max', restore_best_weights=True)


# Train the UNet model
history = unet_model.fit(train_ds.batch(batch_size), validation_data=val_ds.batch(batch_size),
                        batch_size=batch_size, epochs=epochs, shuffle=shuffle, callbacks=callback)

# Plot the performance of the UNet model (Loss vs Dice Loss)
plot_performance_model(history)
plot_performance_loss_model(history)

# Evaluates the model
loss, acc = model.evaluate(test_ds.batch(batch_size), verbose=2)

# Uses the test dataset to test the model on the predicted masks and displays a subset of results
if print_predicted_images:
    show_predicted_images(num_display_examples, model)