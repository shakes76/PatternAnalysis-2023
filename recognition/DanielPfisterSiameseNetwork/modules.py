#%%
import tensorflow as tf
from tensorflow.keras import layers

def siamese_network(height,width,dimension):

    #deifne input of siamese network
    left_input = layers.Input(shape = (height, width, dimension))
    right_input = layers.Input(shape = (height, width, dimension))

    #define standard model of the left and right siamese network
    network = tf.keras.Sequential([layers.Reshape((height, width, dimension)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='sigmoid'),
])

    #save the features from the left and right network in two variables
    feature_vector_left_output = network(left_input)
    feature_vector_right_output = network(right_input)

    #euclidean distance layer
    euclidean_distance = tf.keras.layers.Lambda(lambda features: tf.keras.backend.abs(features[0] - features[1]))([feature_vector_left_output, feature_vector_right_output])

    #fully connected layers
    #dense = layers.Dense(64, activation='relu')(euclidean_distance)
    #output = layers.Dense(1, activation='sigmoid')(dense)
    output = layers.Dense(1, activation='sigmoid')(euclidean_distance)
    #create whole neural network model
    model = tf.keras.Model(inputs=[left_input, right_input], outputs=output)

    model.summary()
    
    #Configurates the loss funciton, optimizer type and metrics of the model for training.
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            metrics=['accuracy'])
    
    return model

# %%
