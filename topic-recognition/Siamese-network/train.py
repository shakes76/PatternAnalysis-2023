# train.py containing the source code for training, validating, testing and saving the model. 
# The model should be imported from “modules.py” and the data loader should be imported from “dataset.py”. 
# Also plot the losses and metrics during training

import tensorflow as tf
import keras.layers as kl
import matplotlib.pyplot as plt
from dataset import data_loader
from modules import siamese_network
from modules import classification_model
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Saving paths
save_trained_snn = '/Users/jollylogan/TryTry/SNN.h5'
save_trained_classifier = '/Users/jollylogan/TryTry/Classifier.h5'

# Load the data
x_train, labels_train, x_val, labels_val, x_test, labels_test, X_data, X_data_labels = data_loader()

def train_and_plot_SNN(epochs=25):

    # Create Siamese model
    model = siamese_network(240, 256)
    model.build(input_shape=(None, 224, 224, 3))

    # Train the model
    siamese_fit = model.fit([x_train[0], x_train[1]], labels_train, validation_data=([x_val[0], x_val[1]], labels_val), epochs=epochs)
    model.save(save_trained_snn)

    # Plot accuracy
    plt.figure()
    plt.plot(siamese_fit.history['accuracy'], label='accuracy')
    plt.plot(siamese_fit.history['val_accuracy'], label='val_accuracy')
    plt.ylim([0.5, 1])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Siamese Model Accuracy') 
    plt.legend(loc='lower right')
    
    # Plot loss
    plt.figure()
    plt.plot(siamese_fit.history['loss'], label='loss')
    plt.plot(siamese_fit.history['val_loss'], label='val_loss')
    plt.ylim([0, 3])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Siamese Model Loss') 
    plt.legend(loc='upper right')

    plt.show()

    return model, siamese_fit

# Run the Siamese model
siamese_model, siamese_fit = train_and_plot_SNN()



def classification_model(cnn) :

    image = kl.Input((240,256,1))

    feature = cnn(image)
    feature = kl.Flatten()(feature)  # Flatten the output before passing to Dense layer
    feature = kl.Dense(units=64, activation="relu")(feature)  # Add more complexity
    feature = kl.BatchNormalization()(feature)

    out = kl.Dense(units = 1, activation= 'sigmoid')(feature)
    classifier = tf.keras.Model([image], out)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001) 

    classifier.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt)
    return classifier

base_network = siamese_model.get_layer("cnn")
base_network.trainable = False

classifier = classification_model(base_network)

# Train the classifier model
labels = tf.reshape(X_data_labels, (-1, 1))

# Define callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

history = classifier.fit(
    X_data, 
    labels, 
    epochs=25, 
    batch_size=36, 
    validation_split=0.2,  # Split 20% of the data for validation
    callbacks=[reduce_lr, early_stopping]  # Use callbacks
)

# Evaluate the classifier
loss, accuracy = classifier.evaluate(X_data, X_data_labels)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Plot accuracy
plt.figure()
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.ylim([0, 1])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Classification Accuracy') 
plt.legend(loc='lower right')

# Plot loss
plt.figure()
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylim([0, 10])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classification Loss') 
plt.legend(loc='upper right')

plt.show()

# Save the trained classifier model
classifier.save(save_trained_classifier)
