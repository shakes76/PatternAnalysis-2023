from tensorflow import keras
import math
import time

import modules as layers
from dataset import DataLoader
from utils import dice_coefficient, dice_loss, DiceCoefficientCallback, plot_accuracy_loss, save_dice_coefficient_plot
from validation import validate_and_visualise_predictions
from predict import test_and_visualise_predictions

# string modifier for saving output files based on time
timestr = time.strftime("%Y%m%d-%H%M%S")
output_dir = "output"

# Constants related to training
EPOCHS = 5
LEARNING_RATE = 0.0005
BATCH_SIZE = 2  # set the batch_size
IMAGE_HEIGHT = 512  # the height input images are scaled to
IMAGE_WIDTH = 512  # the width input images are scaled to
CHANNELS = 3
STEPS_PER_EPOCH_TRAIN = math.floor(2594 / BATCH_SIZE)
STEPS_PER_EPOCH_TEST = math.floor(100 / BATCH_SIZE)


def train_model_check_accuracy(training_data, validation_data):
    model = layers.improved_unet(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
    model.summary()
    # Define the DiceCoefficientCallback
    dice_coefficient_callback = DiceCoefficientCallback(validation_data, STEPS_PER_EPOCH_TEST)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss=dice_loss, metrics=['accuracy', dice_coefficient])
    track = model.fit(
        training_data,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        use_multiprocessing=False,
        callbacks=[dice_coefficient_callback])  # Add the callback her
    plot_accuracy_loss(track, output_dir, timestr)  # Plot accuracy and loss curves

    print("\nEvaluating validation images...")
    validation_loss, validation_accuracy, validation_dice = \
        model.evaluate(validation_data, steps=STEPS_PER_EPOCH_TEST, verbose=2, use_multiprocessing=False)
    print("Validation Accuracy: " + str(validation_accuracy))
    print("Validation Loss: " + str(validation_loss))
    print("Validation DSC: " + str(validation_dice) + "\n")
    
    return model, track.history['dice_coefficient']


# Run the test driver.
def main():
    # Constants related to preprocessing
    train_dir = "datasets/training_input"
    train_groundtruth_dir = "datasets/training_groundtruth"
    validation_dir = "datasets/validation_input"
    validation_groundtruth_dir = "datasets/validation_groundtruth"
    image_mode = "rgb"
    mask_mode = "grayscale"
    image_height = 512
    image_width = 512
    batch_size = 2
    seed = 45
    shear_range = 0.1
    zoom_range = 0.1
    horizontal_flip = True
    vertical_flip = True
    fill_mode = 'nearest'
    number_of_predictions = 3

    print("\nPREPROCESSING IMAGES")
    # print number of images in each directory
    validation_data = DataLoader(
        validation_dir, validation_groundtruth_dir, image_mode, mask_mode, image_height, image_width, batch_size, seed,
        shear_range, zoom_range, horizontal_flip, vertical_flip, fill_mode)
    validation_data = validation_data.create_data_generators()
    train_data = DataLoader(
        train_dir, train_groundtruth_dir, image_mode, mask_mode, image_height, image_width, batch_size, seed,
        shear_range, zoom_range, horizontal_flip, vertical_flip, fill_mode)
    train_data = train_data.create_data_generators()

    print("\nTRAINING MODEL")
    model, dice_history = train_model_check_accuracy(train_data, validation_data)
    # Save Dice coefficient
    save_dice_coefficient_plot(dice_history, output_dir, timestr)
    # Save the trained model to a file

    print("\nSAVING MODEL")
    keras.saving.save_model(model, f"models/my_model_{timestr}.keras", overwrite=True)

    test_dir = "datasets/test_input"
    test_groundtruth_dir = "datasets/test_groundtruth"
    test_data = DataLoader(
        test_dir, test_groundtruth_dir, image_mode, mask_mode, image_height, image_width, batch_size, seed,
        shear_range, zoom_range, horizontal_flip, vertical_flip, fill_mode)
    test_data = test_data.create_data_generators()

    # print("\nVISUALISING PREDICTIONS")
    # validate_and_visualise_predictions(model, validation_data, output_dir, timestr, number_of_predictions)

    test_and_visualise_predictions(model, test_data, output_dir, timestr, number_of_predictions)
    print("COMPLETED")
    return 0


if __name__ == "__main__":
    main()