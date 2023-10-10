import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Input

"""
Model class used to create a Siamese Neural Network

Inspiration taken from 
"https://github.com/schatty/siamese-networks-tf/blob/master/siamnet/models/siamese.py"
"""
class SiameseModel:
    """
    Initialises the model class by calling the create_network() function
    """
    def __init__(self):
        self.model = self.create_network()
    
    """
    Creates the base network shared by the Anchor, Positive and Negative inputs
    Assumes input shape is (256, 240, 3)
    
    Returns:
        base_model - the model to be used for the anchor, positive and negative input
    """
    def base_network(self):
        base_model = Sequential()
        base_model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(256, 240, 3)))
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
        margin - margin between positive and negative pairs, default is 0.1

    triplet loss as defined by "https://en.wikipedia.org/wiki/Triplet_loss"
    """
    def triplet_loss(self, anchor, positive, negative, margin=0.1):
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = pos_dist - neg_dist + margin
        return tf.maximum(0.0, loss)
    
    """
    Class to create the siamese network. Called by  __init__ 

    Outputs:
        siamese_network - the siamese network model compilation
    """
    def create_network(self):
        base_model = self.base_network()

        anchor_input = Input(shape=(256, 240, 3))
        positive_input = Input(shape=(256, 240, 3))
        negative_input = Input(shape=(256, 240, 3))

        anchor_layer = base_model(anchor_input)
        positive_layer = base_model(positive_input)
        negative_layer = base_model(negative_input)

        # Create a dense layer for class prediction using Softmax activation.
        prediction_layer = Dense(2, activation='softmax')(anchor_layer)
    
        siamese_network = Model(inputs=[anchor_input, positive_input, negative_input], outputs=prediction_layer)
        siamese_network.compile(optimizer='adam', loss=self.triplet_loss, metrics=['accuracy'])

        return siamese_network
    
