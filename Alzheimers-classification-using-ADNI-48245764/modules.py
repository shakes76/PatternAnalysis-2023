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


preprocessing = keras.Sequential(
    [
        Normalization(),
        Resizing(image_size, image_size),
        RandomFlip("horizontal"),
        RandomRotation(factor=0.01),
    ],
    name="preprocessing"
)


def perceptron(tensor_in, hidden_units, dropout):
    for units in hidden_units:
        tensor_in = Dense(units, activation=tf.nn.gelu)(tensor_in)
        tensor_in = Dropout(dropout)(tensor_in)
    return tensor_in


class Patches(Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(Layer):
    def __init__(self, patch, dimensions):
        super().__init__()
        self.patch = patch
        self.projection = Dense(units=dimensions)
        self.position_embedding = Embedding(input_dim=patch, output_dim=dimensions)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.patch, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

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
