import tensorflow as tf
import keras.api._v2.keras as keras # Required as, though it violates Python conventions, my TF cannot load Keras properly

from keras.layers import *
from keras.models import Sequential, Model

input_A = Input((240, 256, 3), name='input_A')
input_B = Input((240, 256, 3), name='input_B')

def get_block(depth):
    return Sequential([Conv2D(depth, 3, 1), # 3x3 padding with a stride of 1
                       BatchNormalization(),
                       LeakyReLU()])

DEPTH = 384

cnn = Sequential([Reshape((240, 256, 3)),
                  get_block(DEPTH),
                  get_block(DEPTH * 2),
                  get_block(DEPTH * 4),
                  get_block(DEPTH * 8),
                  get_block(DEPTH * 16),
                  GlobalAveragePooling2D(),
                  Dense(384, activation='sigmoid')])

# As we are using multiple inputs, we concatenate the inputs' feature vectors
feature_vector_A = cnn(input_A)
feature_vector_B = cnn(input_B)
feature_vectors = Concatenate()([feature_vector_A, feature_vector_B])

# Add a Dense layer for non-linearity
dense = Dense(384, activation='sigmoid')(feature_vectors)

# Choose Sigmoid to ensure it's a probability
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[input_A, input_B], outputs = output)

def get_model():
    return model