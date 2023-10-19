import tensorflow as tf
import keras.layers as kl
import keras.backend as kb


def create_siamese_network(height: int, width: int):

    def create_distance_layer(im1_feature, im2_feature):
      tensor = kb.sum(kb.square(im1_feature - im2_feature), axis=1, keepdims=True)
      return kb.sqrt(kb.maximum(tensor, kb.epsilon()))

    # Create a shared neural network (subnetwork)
    subnet = create_cnn_network(height, width)

    # Define inputs for two images
    image1_input = kl.Input((height, width, 1))
    image2_input = kl.Input((height, width, 1))

    # Pass the inputs through the shared subnetwork
    feature1 = subnet(image1_input)
    feature2 = subnet(image2_input)

    # Compute the distance between the feature vectors
    distance = create_distance_layer(feature1, feature2)
    normalized_distance = kl.BatchNormalization()(distance)

    # Output layer for similarity prediction
    output = kl.Dense(units=1, activation='sigmoid')(normalized_distance)

    # Create the Siamese model
    siamese_model = tf.keras.models.Model([image1_input, image2_input], output)

    # Define the optimizer and compile the model with the contrastive loss
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    siamese_model.compile(loss=create_contrastive_loss, metrics=['accuracy'], optimizer=optimizer)

    return siamese_model

def create_cnn_network(height, width):
    # Define the input layer
    input_layer = kl.Input(shape=(height, width, 1))

    # Convolutional and pooling layers for feature extraction
    conv1 = kl.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool1 = kl.MaxPooling2D((2, 2))(conv1)
    conv2 = kl.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = kl.MaxPooling2D((2, 2))(conv2)
    conv3 = kl.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = kl.MaxPooling2D((2, 2))(conv3)

    # Flatten the output
    flat = kl.Flatten()(pool3)

    # Fully connected layers
    dense1 = kl.Dense(512, activation='relu', kernel_regularizer='l2')(flat)
    dense2 = kl.Dense(512, activation='relu', kernel_regularizer='l2')(dense1)

    # Create and return the neural network model
    subnet = tf.keras.models.Model(inputs=input_layer, outputs=dense2, name='subnet')
    return subnet

def create_classification_model(subnet):
    image_input = kl.Input((128, 128, 1))
    feature = subnet(image_input)
    feature = kl.BatchNormalization()(feature)
    output_layer = kl.Dense(units=1, activation='sigmoid')(feature)
    classifier = tf.keras.models.Model([image_input], output_layer)
    optimizer = tf.optimizers.Adam(learning_rate=0.0001)
    classifier.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=optimizer)
    return classifier

# Define the contrastive loss function
def create_contrastive_loss(y, y_pred):
    square = tf.math.square(y_pred)
    margin = tf.math.square(tf.math.maximum(1 - y_pred, 0))
    return tf.math.reduce_mean((1 - y) * square + (y) * margin)