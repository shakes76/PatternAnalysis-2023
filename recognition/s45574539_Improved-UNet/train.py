import dataset
import modules
import tensorflow.keras.backend as be


def dice_similarity(y_true, y_pred):
    """
    Used / based on code from https://notebook.community/cshallue/models/samples/outreach/blogs/segmentation_blogpost/image_segmentation
    (From section: Defining custom metrics and loss functions)
    """
    smooth = 1
    y_true_f = be.flatten(y_true)
    y_pred_f = be.flatten(y_pred)
    intersection = be.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (be.sum(y_true_f) + be.sum(y_pred_f) + smooth)
