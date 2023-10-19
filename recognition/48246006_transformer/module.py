import tensorflow as tf
import tensorflow_hub as hub

def transformer_model(input_shape):
    model_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2"
    hub_layer = hub.KerasLayer(model_url, trainable=True)

    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = hub_layer(input_layer)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    return model
