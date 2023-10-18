import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

class SiameseModel:
    def __init__(self):
        self.model = self.create_network()

    def base_network(self):
        input_image = Input(shape=(256, 240, 3))
        x = Conv2D(64, (3, 3), activation='relu')(input_image)
        x = MaxPooling2D()(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        return Model(input_image, x)

    def triplet_loss(self, y_true, y_pred, alpha=0.2):
        anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
        loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
        return tf.reduce_mean(loss)

    def create_network(self):
        anchor_input = Input(shape=(256, 240, 3))
        positive_input = Input(shape=(256, 240, 3))
        negative_input = Input(shape=(256, 240, 3))

        base_model = self.base_network()

        anchor_embedding = base_model(anchor_input)
        positive_embedding = base_model(positive_input)
        negative_embedding = base_model(negative_input)

        inputs = [anchor_input, positive_input, negative_input]

        # Output the embeddings as a list
        output_embeddings = [anchor_embedding, positive_embedding, negative_embedding]

        siamese_network = Model(inputs=inputs, outputs=output_embeddings)
        siamese_network.compile(optimizer='adam', loss=tfa.losses.TripletSemiHardLoss())
        return siamese_network
    
    def create_classifier(self):
        input_image = Input(shape=(256, 240, 3))

        # Reuse the same base network architecture as the Siamese model
        base_model = self.base_network()
        classifier_embedding = base_model(input_image)

        # Add additional layers for classification
        classifier_output = Dense(2, activation='softmax')(classifier_embedding)

        # Create the classifier model
        classifier_model = Model(inputs=input_image, outputs=classifier_output)
        return classifier_model
    
