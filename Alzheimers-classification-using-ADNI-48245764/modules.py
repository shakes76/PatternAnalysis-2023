import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Layer, Flatten, Normalization, Resizing, Dropout, Embedding, RandomFlip, RandomRotation, RandomZoom, Input, LayerNormalization, MultiHeadAttention, Add
import os    

num_classes = 2
image_size = 140
batch_size = 140
patch_size = 16
dimensions = 96
num_heads = 4
patch = (image_size // patch_size) ** 2
vit_dim = [dimensions * 2, dimensions, ]
vit_layer = 5
mlp_units = [2048, 1024]
input_shape = (140, 140, 3)


def perceptron(tensor_in, hidden_units, dropout):
    for units in hidden_units:
        tensor_in = Dense(units, activation=tf.nn.gelu)(tensor_in)
        tensor_in = Dropout(dropout)(tensor_in)
    return tensor_in


def transformer():
    inputs = Input(shape=input_shape)
    augmented_inputs = preprocessing(inputs)
    patches = Patches(patch_size)(augmented_inputs)
    encoded_patches = PatchEncoder(patch, dimensions)(patches)

    for _ in range(vit_layer):
        
        Layer1 = LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = MultiHeadAttention(num_heads=num_heads,key_dim=dimensions,dropout=0.1)
        (Layer1, Layer1)

        Layer2 = Add()([attention_output, encoded_patches])
        
        Layer3 = LayerNormalization(epsilon=1e-6)(Layer2)
        Layer3 = perceptron(Layer3,hidden_units=vit_dim,dropout=0.1)

        encoded_patches = Add()([Layer3, Layer2])

    output = LayerNormalization(epsilon=1e-6)(encoded_patches)
    output = Flatten()(output)
    output = Dropout(0.5)(output)

    features = perceptron(output, hidden_units=mlp_units, dropout=0.5)

    logits = Dense(num_classes)(features)
    model = keras.Model(inputs=inputs, outputs=logits)

    return model
