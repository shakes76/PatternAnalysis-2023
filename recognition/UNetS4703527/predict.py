import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import accuracy_score, precision_score
from dataset import load_data
from utils import dice_coef, read_image_predict, read_mask_predict, save_results, parse_args

# Inisalise values
split = 0.2
SCORE = []

def predict_mask(model, test_x, test_y):
    
    for x, y in tqdm(zip(test_x, test_y), total=len(test_x)):
        # Grab the img name and set img path
        name = x.split("/")[-1]
        name = os.path.basename(name)
        save_image_path = os.path.join("results", name)

        image, x = read_image_predict(x)
        mask, y = read_mask_predict(y)

        # Start model predicton
        y_pred = model.predict(x)[0] > 0.5
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred.astype(np.int32)

        # Save results to the img path
        save_results(image, mask, y_pred, save_image_path)

        # Flatten to calculate the dice accuracy and precision
        y = y.flatten()
        y_pred = y_pred.flatten()
        acc_value = accuracy_score(y, y_pred)
        precision_value = precision_score(y, y_pred, labels=[0, 1], average="binary")
        dice_value = dice_coef(y, y_pred)
        
        # Add the scored to the Score list
        SCORE.append([name, acc_value, precision_value, dice_value])

if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.path

    # Load the saved model from files
    with CustomObjectScope({'dice_coef': dice_coef}):
        model = tf.keras.models.load_model("files/model.h5")
    # Load the testing data
    _, _, (test_x, test_y) = load_data(dataset_path, split)

    predict_mask(model, test_x, test_y)

    # Calculate mean scores
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"Precision: {score[1]:0.5f}")
    print(f"Dice Coefficient: {score[2]:0.5f}")
    
    # Save scores to a csv file 
    df = pd.DataFrame(SCORE, columns = ["Image Name", "Acc", "Precision", "Dice Coefficient"])
    df.to_csv("files/score.csv")

    