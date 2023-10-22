import dataset
import modules
import tensorflow as tf
import tensorflow.keras.backend as be


def dice_similarity(y_true, y_pred):
    """
    Based on code from https://notebook.community/cshallue/models/samples/outreach/blogs/segmentation_blogpost/image_segmentation
    (From section: Defining custom metrics and loss functions)
    """
    smooth = 1
    y_true_f = be.flatten(y_true)
    y_pred_f = be.flatten(y_pred)
    intersection = be.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (be.sum(y_true_f) + be.sum(y_pred_f) + smooth)


def train_model():
    train_data, test_data, val_data = dataset.load_data()
    model = modules.improved_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_similarity])
    history = model.fit(train_data.batch(10), validation_data=val_data.batch(10), epochs=10)
    model.evaluate(test_data.batch(10))


if __name__ == "__main__":
    train_model()
