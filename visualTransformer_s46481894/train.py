# imports
import datasets
import tensorflow as tf
from tensorflow import keras

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
img_size = 256
batch_size=128



def train(model):
    optimiser = tf.optimizers.Adam(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    model.compile(
        optimizer=optimiser,
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(5, name="top 5 accuracy"),
        ],
    )



