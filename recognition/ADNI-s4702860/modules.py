from tensorflow import keras
from tensorflow.keras import layers
from dataset import load_dataset
from tensorflow.keras import layers, models


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.keras.layers import Dense, Flatten, Conv2D, BatchNormalization, MaxPool2D, ReLU
from tensorflow.keras import Model

"""
Model class used to create a Siamese Neural Network

Inspiration taken from 
"https://github.com/schatty/siamese-networks-tf/blob/master/siamnet/models/siamese.py"
"""
class Model:
    """
    Initialises the model class
    """
    def __init__(self):
        pass
    
    """
    Creates the base network shared by the Anchor, Positive and Negative inputs

    Inputs:
        input_shape - the size of the input shape
    
    Returns:
        base_model - the model to be used for the anchor, positive and negative input
    """
    def base_network(self, input_shape):
        base_model = Sequential() # Defines the base model
        base_model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
        base_model.add(MaxPool2D())
        base_model.add(Conv2D(128, (3, 3), activation='relu'))
        base_model.add(MaxPool2D())
        base_model.add(Flatten())
        base_model.add(Dense(256, activation='relu'))
        base_model.add(Dense(128, activation='relu'))
        return base_model
    
    """
    Function used to define the triplet loss

    Inputs:
        anchor - the anchor image
        positive - image the same class as the anchor
        negative - image the opposite class of the anchor
    """
    def triplet_loss(self, anchor, positive, negative):
        pass
