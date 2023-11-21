import math
import time

from tensorflow import keras

import modules as layers
from dataset import DataLoader
from utils import (
    dice_coefficient,
    dice_loss,
    DiceCoefficientCallback,
    plot_accuracy_loss,
    save_dice_coefficient_plot
)
from validation import validate_and_visualise_predictions
from predict import test_and_visualise_predictions

# Constants
EPOCHS = 5
LEARNING_RATE = 0.0005
BATCH_SIZE = 2
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
CHANNELS = 3
STEPS_PER_EPOCH_TRAIN = math.floor(2594 / BATCH_SIZE)
STEPS_PER_EPOCH_TEST = math.floor(100 / BATCH_SIZE)

# String modifier for saving output files based on time
timestr = time.strftime("%Y%m%d-%H%M%S")
output_dir = "output"


def train_model_check_accuracy(training_data, validation_data):
    model = layers.improved_unet(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)
    model.summary()

    dice_coefficient_callback = DiceCoefficientCallback(validation_data, STEPS_PER_EPOCH_TEST)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE),
                  loss=dice_loss,
                  metrics=['accuracy', dice_coefficient])

    track = model.fit(
        training_data,
        steps_per_epoch=STEPS_PER_EPOCH_TRAIN,
        epochs=EPOCHS,
        shuffle=True,
        verbose=1,
        use_multiprocessing=False,
        callbacks=[dice_coefficient_callback]
    )

    plot_accuracy_loss(track, output_dir, timestr)

    print("\nEvaluating validation images...")
    validation_loss, validation_accuracy, validation_dice = \
        model.evaluate(validation_data, steps=STEPS_PER_EPOCH_TEST, verbose=2, use_multiprocessing=False)
    
    print(f"Validation Accuracy: {validation_accuracy}")
    print(f"Validation Loss: {validation_loss}")
    print(f"Validation DSC: {validation_dice}\n")
    
    return model, track.history['dice_coefficient']


def main():
    preprocessing_params = {
        "image_mode": "rgb",
        "mask_mode": "grayscale",
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "batch_size": BATCH_SIZE,
        "seed": 45,
        "shear_range": 0.1,
        "zoom_range": 0.1,
        "horizontal_flip": True,
        "vertical_flip": True,
        "fill_mode": 'nearest'
    }

    print("\nPREPROCESSING IMAGES")
    validation_data = DataLoader(
        "datasets/validation_input",
        "datasets/validation_groundtruth",
        **preprocessing_params
    ).create_data_generators()

    train_data = DataLoader(
        "datasets/training_input",
        "datasets/training_groundtruth",
        **preprocessing_params
    ).create_data_generators()

    print("\nTRAINING MODEL")
    model, dice_history = train_model_check_accuracy(train_data, validation_data)
    save_dice_coefficient_plot(dice_history, output_dir, timestr)

    print("\nSAVING MODEL")
    keras.saving.save_model(model, f"models/my_model_{timestr}.keras", overwrite=True)

    test_data = DataLoader(
        "datasets/test_input",
        "datasets/test_groundtruth",
        **preprocessing_params
    ).create_data_generators()

    number_of_predictions = 3
    test_and_visualise_predictions(model, test_data, output_dir, timestr, number_of_predictions)

    print("COMPLETED")
    return 0


if __name__ == "__main__":
    main()
