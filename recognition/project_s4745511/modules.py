import tensorflow.keras as k
import keras.layers as kl
import keras.backend as kb
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# def cnn_network(height, width):
#     model = Sequential()

#     # Block 1
#     model.add(Conv2D(64, (3, 3), input_shape=(height, width, 1), padding="same", activation="relu"))
#     model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     # Block 2
#     model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     # Block 3
#     model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

#     # Fully Connected Layers
#     model.add(Flatten())
#     model.add(Dense(4096, activation='relu'))
#     model.add(Dense(4096, activation='relu'))
#     # model.add(Dense(num_classes, activation='softmax'))

#     return model


# import tensorflow as tf
# from tensorflow.keras import layers as kl
# from tensorflow.keras import Model as k

def cnn_network(height, width):
    """ VGG16-like network for grayscale images

    Args:
        height (int): Height of the input image
        width (int): Width of the input image

    Returns:
        tf.keras.Model: The VGG16-like model for grayscale images
    """
    input = kl.Input(shape=(height, width, 1))  # Input has 1 channel for grayscale image

    # Block 1
    conv1_1 = kl.Conv2D(64, (3, 3), activation='relu', padding='same')(input)
    conv1_2 = kl.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1_1)
    pool1 = kl.MaxPooling2D((2, 2))(conv1_2)

    # Block 2
    conv2_1 = kl.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2_2 = kl.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2_1)
    pool2 = kl.MaxPooling2D((2, 2))(conv2_2)

    # Block 3
    conv3_1 = kl.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3_2 = kl.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_1)
    conv3_3 = kl.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3_2)
    pool3 = kl.MaxPooling2D((2, 2))(conv3_3)

    # Block 4
    conv4_1 = kl.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4_2 = kl.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4_1)
    conv4_3 = kl.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4_2)
    pool4 = kl.MaxPooling2D((2, 2))(conv4_3)

    # Block 5
    conv5_1 = kl.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5_2 = kl.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5_1)
    conv5_3 = kl.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5_2)
    pool5 = kl.MaxPooling2D((2, 2))(conv5_3)

    # Flatten and Fully Connected Layers
    flat = kl.Flatten()(pool5)
    dense1 = kl.Dense(4096, activation='relu')(flat)
    dense2 = kl.Dense(4096, activation='relu')(dense1)
    output = kl.Dense(2, activation='softmax')(dense2)  # NUM_CLASSES is the number of output classes

    model = k.Model(inputs=input, outputs=output, name='subnet')

    return model


# def cnn_network(height, width):
#     """ The modified subnetwork using CNN

#     Args:
#         height (int): Height of the input image
#         width (int): Width of the input image

#     Returns:
#         tf.keras.Model: The modified subnetwork Model
#     """
#     input = kl.Input(shape=(height, width, 1))
#     conv1 = kl.Conv2D(32, (3, 3), activation='relu', padding='same')(input)
#     pool1 = kl.MaxPooling2D((2, 2))(conv1)
#     conv2 = kl.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
#     pool2 = kl.MaxPooling2D((2, 2))(conv2)
#     conv3 = kl.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
#     pool3 = kl.MaxPooling2D((2, 2))(conv3)
#     flat = kl.Flatten()(pool3)
#     dense1 = kl.Dense(512, activation='relu', kernel_regularizer='l2')(flat)
#     dense2 = kl.Dense(512, activation='relu', kernel_regularizer='l2')(dense1)

#     subnet = k.Model(inputs=input, outputs=dense2, name='subnet')

#     return subnet


def distance_layer(im1_feature, im2_feature):
    """ Layer to compute (euclidean) difference between feature vectors

    Args:
        im1_feature (tensor): feature vector of an image
        im2_feature (tensor): feature vector of an image

    Returns:
        tensor: Tensor containing differences
    """
    tensor = kb.sum(kb.square(im1_feature - im2_feature), axis=1, keepdims=True)
    return kb.sqrt(kb.maximum(tensor, kb.epsilon())) 


def classification_model(subnet):
    """ Build the classification Model with a CNN-based subnetwork

    Args:
        subnet (layer): the modified subnetwork with convolutional layers

    Returns:
        model: compiled model
    """
    image = kl.Input((128, 128, 1))
    tensor = subnet(image)
    tensor = kl.BatchNormalization()(tensor)
    out = kl.Dense(units=1, activation='sigmoid')(tensor)

    classifier = Model([image], out)

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    classifier.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)

    return classifier


def contrastive_loss(y, y_pred):
    """
    Sourced from [Siameses Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive/)
    """
    square = tf.math.square(y_pred)
    margin = tf.math.square(tf.math.maximum(1 - (y_pred), 0))
    return tf.math.reduce_mean((1 - y) * square + (y) * margin)


def siamese(height: int, width: int):
    """ The SNN. Passes image pairs through the subnetwork,
        and computes distance between output vectors. 

    Args:
        height (int): height of input image
        width (int): width of input image

    Returns:
        Model: compiled model
    """

    subnet = cnn_network(height, width)


    image1 = kl.Input((height, width, 1))
    image2 = kl.Input((height, width, 1))

    feature1 = subnet(image1)
    feature2 = subnet(image2)

    distance = distance_layer(feature1, feature2)

    # Classification
    tensor = kl.BatchNormalization()(distance)
    out = kl.Dense(units = 1, activation='sigmoid')(tensor)

    model = Model([image1, image2], out)

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss=contrastive_loss, metrics=['accuracy'],optimizer=opt)

    return model