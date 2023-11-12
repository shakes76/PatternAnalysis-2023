# imports
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Layer, Flatten, Normalization, Resizing, RandomFlip, RandomRotation, \
    RandomZoom, Dropout, Embedding, Input, LayerNormalization, MultiHeadAttention, Add

# set parameters
num_classes = 2
img_size = 128
batch_size = 128
patch_size = 16
num_patches = (img_size // patch_size) ** 2
projection_dim = 96
num_heads = 4
transformer_units = [projection_dim * 2, projection_dim, ]
transformer_layers = 5
mlp_head_units = [2048, 1024]

input_shape = (128, 128, 3)  # may need to change size

# data augmentation is used to modify data to create more datapoints
data_augmentation = keras.Sequential(
    [
        Normalization(),
        Resizing(img_size, img_size),
        RandomFlip("horizontal"),
        RandomRotation(factor=0.02),
        RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation"
)


# implement multi layer perceptron
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x


# create Patches
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


# project a patch to a vector of size projection_dim
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def create_vit():
    inputs = Input(shape=input_shape)
    augmented = data_augmentation(inputs)  # augment the data
    patches = Patches(patch_size)(augmented)  # create patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)  # encode patches

    # make transformer layers
    for _ in range(transformer_layers):
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=0.1
        )(x1, x1)

        x2 = Add()([attention_output, encoded_patches])

        x3 = LayerNormalization(epsilon=1e-6)(x2)

        x3 = mlp(x3,
                 hidden_units=transformer_units,
                 dropout_rate=0.1
                 )

        encoded_patches = Add()([x3, x2])

    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)

    logits = Dense(num_classes)(features)
    # make the model with keras
    model = keras.Model(inputs=inputs, outputs=logits)

    return model
