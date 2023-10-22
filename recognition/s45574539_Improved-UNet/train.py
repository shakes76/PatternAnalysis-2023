import dataset
import modules
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as be


def dice_similarity(y_true, y_pred):
    """
    Function to calculate the dice similarity coefficient.
    Based on code from https://notebook.community/cshallue/models/samples/outreach/blogs/segmentation_blogpost/image_segmentation
    (From section: Defining custom metrics and loss functions)
    """
    smooth = 1
    y_true_f = be.flatten(y_true)
    y_pred_f = be.flatten(y_pred)
    intersection = be.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (be.sum(y_true_f) + be.sum(y_pred_f) + smooth)


def train_model():
    # Use data loader from dataset.py
    train_data, test_data, val_data = dataset.load_data()

    # Use model from modules.py
    model = modules.improved_unet()

    # Compile improved UNet model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_similarity])
    history = model.fit(train_data.batch(10), validation_data=val_data.batch(10), epochs=10)

    # Plot dice similarity of the model
    plot_dice_similarity(history)

    # evaluate on test dataset
    model.evaluate(test_data.batch(10))


def plot_dice_similarity(history):
    """
    Plots the dice similarity coefficient readings for the trained model.
    :param history: history of the trained model
    :return: image / plot of dice similarity
    """
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['dice_similarity'], label='Training')
    plt.plot(history.history['val_dice_similarity'], label='Validation')
    plt.legend()
    plt.title('Dice Similarity Coefficient Test vs Validation')
    plt.xlabel('# of Epochs')
    plt.ylabel('Dice Similarity Coefficient')
    plt.savefig('dice_similarity.png')
    plt.show()


# run main function
if __name__ == "__main__":
    train_model()
