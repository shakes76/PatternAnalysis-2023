import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

"""
Class to handle both the siamese model and the classifier. 

Architecture (input linking) was influenced by the video 
below for a 2 input siamese neural network
https://youtu.be/DGJyh5dK4hU?si=2LxFEx1aOVHJEFBu
"""
class Modules():
    """
    Initialises the Modules class
    """
    def __init__(self):
        pass

    """
    Creates the base model architecture that will be used for the siamese architecture

    outputs:
        model - the base network used for the layers of the siamese model
    """
    def base_network(self):
        input_image = Input(shape=(256, 240, 3))
        x = Conv2D(64, (3, 3), activation='relu')(input_image)
        x = BatchNormalization()(x)  # Add BatchNormalization after Conv2D
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization after Conv2D
        x = MaxPooling2D()(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization after Dense
        x = Dropout(0.25)(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)  # Add BatchNormalization after Dense
        x = Dropout(0.25)(x)
        model = Model(input_image, x)
        return model

    """
    Function that creates a siamese neural network with three inputs.
    The model has anchor (neutral class), positive (same class as anchor), 
    negative (opposite class as anchor), and returns the embeddings of the anchor. 

    Outputs:
        siamese_network = A compiled siamese network ready for training
    """
    def create_siamese_network(self):
        # create the input layers
        anchor_input = Input(shape=(256, 240, 3))
        positive_input = Input(shape=(256, 240, 3))
        negative_input = Input(shape=(256, 240, 3))

        # create same model for inputs to share weights
        base_model = self.base_network()
        anchor_embedding = base_model(anchor_input)
        positive_embedding = base_model(positive_input)
        negative_embedding = base_model(negative_input)

        # define model architecture
        inputs = [anchor_input, positive_input, negative_input]
        outputs = [anchor_embedding, positive_embedding, negative_embedding]
        siamese_network = Model(inputs=inputs, outputs=outputs)
        siamese_network.compile(optimizer='adam', loss=tfa.losses.TripletSemiHardLoss())
        return siamese_network
    
    """
    Function that creates a classifier based on the siamese embeddings.

    Outputs:
        classifier_model - A compiled siamese network ready for training.
    """
    def create_classifier(self):
        anchor_embedding_input = Input(shape=(128,))  # Assuming the embedding size is 128
        x = Dense(64, activation='relu')(anchor_embedding_input)
        x = Dense(32, activation='relu')(x)
        output = Dense(2, activation='softmax')(x)  # Adjust num_classes accordingly
        classifier_model = Model(inputs=anchor_embedding_input, outputs=output)
        classifier_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return classifier_model
    
