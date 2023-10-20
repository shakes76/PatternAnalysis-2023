import tensorflow as tf
from module import build_alzheimer_model

# Load the trained model
model = tf.keras.models.load_model('alzheimer_model.h5')

# Load and preprocess new data (if needed)
# new_data = load_and_preprocess_new_data()

# Make predictions
# predictions = model.predict(new_data)
