import tensorflow as tf
from module import build_alzheimer_model

# Load the trained model
model = tf.keras.models.load_model('alzheimer_model.h5')
