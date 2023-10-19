import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope


H = 256
W = 256
split = 0.2
dataset_path = r"C:\Users\raulm\Desktop\Uni\Sem2.2023\Patterns\ISIC-2017_Training_Data"
SCORE = []

if __name__ == "__main__":

    # Load the saved model from files
    with CustomObjectScope({'dice_coef': dice_coef_predict}):
        model = tf.keras.models.load_model("files/model.h5")
    # Load the testing data
    _, _, (test_x, test_y) = load_data(dataset_path, split)

    