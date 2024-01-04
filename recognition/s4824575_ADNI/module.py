import tensorflow as tf

class Embedder(tf.keras.layers.Layer):

    def __init__(self, patch_size, dim):
        super(Embedder, self).__init__()

        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(dim, kernel_size=patch_size, strides=patch_size),
            tf.keras.layers.BatchNormalization()
        ])

    def call(self, inputs):
        encoded = self.encoder(inputs)
        flattened = tf.reshape(encoded, [-1, tf.shape(encoded)[1] * tf.shape(encoded)[2], encoded.shape[-1]])
        return flattened

class VisionModel(tf.keras.Model):

    def __init__(self, num_classes, image_size, patch_size, num_patches, dim, depth, num_heads, mlp_dim, **kwargs):
        super(VisionModel, self).__init__()

        self.embedder = Embedder(patch_size, dim)

        self.token = tf.Variable(tf.random.normal((1, 1, dim)))
        self.positions = tf.Variable(tf.random.normal((1, num_patches + 1, dim)))

        self.transformer = tf.keras.layers.Transformer(
            num_layers=depth,
            num_heads=num_heads,
            d_model=dim,
            mlp_dim=mlp_dim,
            dropout=0.1,
            name="transformer"
        )

        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dense(dim),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, inputs):
        embedded = self.embedder(inputs)

        token = tf.repeat(self.token, embedded.shape[0], axis=0)
        embedded = tf.concat([token, embedded], axis=1)
        embedded += self.positions

        encoded = self.transformer(embedded)
        aggregated = tf.reduce_mean(encoded, axis=1)

        return self.mlp_head(aggregated)



    

    
        
    