import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import os
import numpy as np
from tqdm import tqdm

# Inisalise values
H = 256
W = 256
split = 0.2
dataset_path = r"C:\Users\raulm\Desktop\Uni\Sem2.2023\Patterns\ISIC-2017_Training_Data"

def predict_mask(model, test_x, test_y):
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        # Grab the img name and set img path
        name = x.split("/")[-1]
        name = os.path.basename(name)
        save_image_path = os.path.join("results", name)

        ori_x, x = read_image_predict(x)
        ori_y, y = read_mask_predict(y)

        # Start model predicton
        y_pred = model.predict(x)[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        # Save results to the img path
        save_results(ori_x, ori_y, y_pred, save_image_path)

        # Flatten to calculate the dice accuracy and precision
        y = y.flatten()
        y_pred = y_pred.flatten()
        acc_value = accuracy_score(y, y_pred)
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        dice_value = dice_coef(y, y_pred)
        
        # Add the scored to the Score list
        SCORE.append([name, acc_value, precision_value, dice_value])

if __name__ == "__main__":

    # Load the saved model from files
    with CustomObjectScope({'dice_coef': dice_coef_predict}):
        model = tf.keras.models.load_model("files/model.h5")
    # Load the testing data
    _, _, (test_x, test_y) = load_data(dataset_path, split)

    predict_mask(model, test_x, test_y)

    