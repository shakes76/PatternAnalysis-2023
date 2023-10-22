from dataset import *
from modules import *
from train import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import saving
import numpy as np
from PIL import Image

#TODO: Save images.
#      Print graph of training history?
#      Write specs etc.

def predict_main():
    #Load in the test batches.
    _, _, test_batches = preprocessing()
    #Unzip the test data to evaluate it.
    #Code used from:
    #https://stackoverflow.com/questions/53641920/tensorflow-tf-data-dataset-api-dataset-unzip-function
    #test_images = test_batches.map(lambda a, b: a)
    #test_masks = test_batches.map(lambda a, b: b)
    #Load model and evaluate on test set.
    cos = {'dice_coef': dice_coef, 'Dice': Dice, 'dice_acc': dice_acc}
    with tf.keras.utils.custom_object_scope(cos):
        iunet_model = tf.keras.saving.load_model("/home/Student/s4428306/report/iunet_model.keras")
    evaluation = iunet_model.evaluate(test_batches, batch_size=64, return_dict=True)
    for name, value in evaluation.items():
        print(f"{name}: {value:.4f}")
    #Save 2 examples for inspection.
    test_data = test_batches.unbatch()
    test_data = list(test_data.as_numpy_iterator())
    #TODO: Refactor this.
    #Process first example.
    ex1_ind = 10
    dp1_image_og = test_data[ex1_ind][0]
    dp1_image = dp1_image_og * 255
    #dp1_image = dp1_image.numpy()
    dp1_image = dp1_image.astype(np.uint8)
    Image.fromarray(dp1_image).save("/home/Student/s4428306/report/example1_image.png")
    dp1_mask = test_data[ex1_ind][1]
    #dp1_mask = np.argmax(dp1_mask, axis=2)
    #dp1_mask = dp1_mask.numpy()
    print("DEBUG")
    print(dp1_mask.shape)
    print(type(dp1_mask))
    dp1_mask = dp1_mask.astype(np.uint8) * 255
    print("DEBUG")
    print(dp1_mask.shape)
    print(type(dp1_mask))
    dp1_mask = np.squeeze(dp1_mask)
    Image.fromarray(dp1_mask).save("/home/Student/s4428306/report/example1_mask.png")
    dp1_pred = iunet_model.call(tf.expand_dims(tf.convert_to_tensor(dp1_image_og), axis=0))
    #dp1_pred = tf.squeeze(dp1_pred)
    print("DP1_PRED SHAPE BEFORE ARGMAX", dp1_pred.shape)
    #dp1_pred = np.argmax(dp1_pred, axis=2)
    dp1_pred = np.round(dp1_pred)
    print("DP1_PRED SHAPE AFTER ARGMAX", dp1_pred.shape)
    #dp1_pred = dp1_pred.numpy()
    dp1_pred = dp1_pred.astype(np.uint8) * 255
    dp1_pred = np.squeeze(dp1_pred)
    Image.fromarray(dp1_pred).save("/home/Student/s4428306/report/example1_pred.png")
    #Process second example.
    ex2_ind = 70
    dp2_image_og = test_data[ex2_ind][0]
    dp2_image = dp2_image_og * 255
    #dp2_image = dp2_image.numpy()
    dp2_image = dp2_image.astype(np.uint8)
    Image.fromarray(dp2_image).save("/home/Student/s4428306/report/example2_image.png")
    dp2_mask = test_data[ex2_ind][1]
    #dp2_mask = np.argmax(dp2_mask, axis=2)
    #dp2_mask = dp2_mask.numpy()
    dp2_mask = dp2_mask.astype(np.uint8) * 255
    dp2_mask = np.squeeze(dp2_mask)
    Image.fromarray(dp2_mask).save("/home/Student/s4428306/report/example2_mask.png")
    dp2_pred = iunet_model.call(tf.expand_dims(tf.convert_to_tensor(dp2_image_og), axis=0))
    #dp2_pred = tf.squeeze(dp2_pred)
    #dp2_pred = np.argmax(dp2_pred, axis=2)
    dp2_pred = np.round(dp2_pred)
    #dp2_pred = dp2_pred.numpy()
    dp2_pred = dp2_pred.astype(np.uint8) * 255
    dp2_pred = np.squeeze(dp2_pred)
    Image.fromarray(dp2_pred).save("/home/Student/s4428306/report/example2_pred.png")

"""
LOAD MODEL
RUN PREDICTIONS ON TEST
PLOT PREDICTIONS METRICS FROM TEST
CHOOSE 2 RANDOM (IMAGE, MASK) PAIRS AND SHOW PREDICTION FOR THEM. (NEED TO CONVERT TO IMAGE.)
"""

if __name__ == "__main__":
    predict_main()

