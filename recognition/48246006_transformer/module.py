import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers
import math

class PatchLayer(Layer):
    def __init__(self, image_size, patch_size, num_patches, projection_dim, **kwargs):
        super(PatchLayer, self).__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.half_patch = patch_size // 2
        self.flatten_patches = layers.Reshape((num_patches, -1))
        self.projection = layers.Dense(units=projection_dim)
        self.layer_norm = layers.LayerNormalization(epsilon=1e-6)

    def shift_images(self, images, mode):
        if mode == 'left-up':
            crop_height, crop_width, shift_height, shift_width = self.half_patch, self.half_patch, 0, 0
        elif mode == 'left-down':
            crop_height, crop_width, shift_height, shift_width = 0, self.half_patch, self.half_patch, 0
        elif mode == 'right-up':
            crop_height, crop_width, shift_height, shift_width = self.half_patch, 0, 0, self.half_patch
        else:
            crop_height, crop_width, shift_height, shift_width = 0, 0, self.half_patch, self.half_patch

        crop = tf.image.crop_to_bounding_box(
            images,
            offset_height=crop_height,
            offset_width=crop_width,
            target_height=self.image_size - self.half_patch,
            target_width=self.image_size - self.half_patch
        )

        shift_pad = tf.image.pad_to_bounding_box(
            crop,
            offset_height=shift_height,
            offset_width=shift_width,
            target_height=self.image_size,
            target_width=self.image_size
        )
        return shift_pad

    def call(self, images):
        images = tf.concat(
            [images,
             self.shift_images(images, mode='left-up'),
             self.shift_images(images, mode='left-down'),
             self.shift_images(images, mode='right-up'),
             self.shift_images(images, mode='right-down')],
            axis=-1
        )
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        flat_patches = self.flatten_patches(patches)
        tokens = self.layer_norm(flat_patches)
        tokens = self.projection(tokens)
        return (tokens, patches)

    def get_config(self):
        config = super(PatchLayer, self).get_config()
        config.update(
            {
                'image_size': self.image_size,
                'patch_size': self.patch_size,
                'num_patches': self.num_patches,
                'projection_dim': self.projection_dim
            }
        )
        return config

class EmbedPatch(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(EmbedPatch, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.position_embedding = layers.Embedding(
            input_dim=self.num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(0, self.num_patches, delta=1)
        return patches + self.position_embedding(positions)

    def get_config(self):
        config = super(EmbedPatch, self).get_config()
        config.update(
            {
                'num_patches': self.num_patches,
                'projection_dim': self.projection_dim
            }
        )
        return config

class MultiHeadAttentionLSA(layers.MultiHeadAttention):
    def __init__(self, **kwargs):
        super(MultiHeadAttentionLSA, self).__init__(**kwargs)
        self.tau = tf.Variable(math.sqrt(float(self._key_dim)), trainable=True)

    def _compute_attention(self, query, key, value, attention_mask=None, training=None):
        query = tf.multiply(query, 1.0/self.tau)
        attention_scores = tf.einsum(self._dot_product_equation, key, query)
        attention_mask = tf.convert_to_tensor(attention_mask)
        attention_scores = self._masked_softmax(attention_scores, attention_mask)
        attention_scores_dropout = self._dropout_layer(
            attention_scores, training=training
        )
        attention_output = tf.einsum(
            self._combine_equation, attention_scores_dropout, value
        )
        return attention_output, attention_scores

    def get_config(self):
        config = super(MultiHeadAttentionLSA, self).get_config()
        return config

def build_vision_transformer(input_shape, image_size, patch_size, num_patches, attention_heads, projection_dim, hidden_units, dropout_rate, transformer_layers, mlp_head_units):
    inputs = layers.Input(shape=input_shape)
    (tokens, _) = PatchLayer(image_size, patch_size, num_patches, projection_dim)(inputs)
    encoded_patches = EmbedPatch(num_patches, projection_dim)(tokens)
    for _ in range(transformer_layers):
        layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        diag_attn_mask = 1 - tf.eye(num_patches)
        diag_attn_mask = tf.cast([diag_attn_mask], dtype=tf.int8)
        attention_output = MultiHeadAttentionLSA(num_heads=attention_heads, key_dim=projection_dim, dropout=dropout_rate)(layer_norm_1, layer_norm_1, attention_mask=diag_attn_mask)
        skip_1 = layers.Add()([attention_output, encoded_patches])
        layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)(skip_1)
        mlp_layer = layer_norm_2
        for units in hidden_units:
            mlp_layer = layers.Dense(units, activation=tf.nn.gelu)(mlp_layer)
            mlp_layer = layers.Dropout(dropout_rate)(mlp_layer, training=False)
        encoded_patches = layers.Add()([mlp_layer, skip_1])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(dropout_rate)(representation, training=False)
    features = representation
    for units in mlp_head_units:
        features = layers.Dense(units, activation=tf.nn.gelu)(features)
        features = layers.Dropout(dropout_rate)(features, training=False)
    logits = layers.Dense(1)(features)
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model
