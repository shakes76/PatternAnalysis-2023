import tensorflow as tf
import matplotlib.pyplot as plt
from modules import create_siamese_network
from modules import create_classification_model
from modules import create_contrastive_loss
from dataset import siamese_data_loader
from dataset import classification_data_loader
from tensorflow.keras.models import load_model

# Define file paths for saving models
SNN_PATH = '/SNN.h5'
CLASSIFIER_PATH = '/Classifier.h5'

def training(epochs_snn=40, epochs_classifier=20):
    """
    Function: training the siamese model and classifier and plotting the accuracy and loss
    Input Parameters:
    - epochs_snn (int): Number of training epochs for the Siamese Network (default is 1).
    - epochs_classifier (int): Number of training epochs for the Classification Model (default is 1).
    """
    # Load the Siamese model data
    siamese_train, siamese_val = siamese_data_loader()

    # Create and compile the Siamese Network
    snn_model = create_siamese_network(128, 128)

    # Train the Siamese Network
    snn_fit = snn_model.fit(siamese_train, epochs=epochs_snn, validation_data=siamese_val)

    # Save the trained Siamese Network model
    snn_model.save(SNN_PATH)

    # Load the Classification Model data
    classify_train, classify_val = classification_data_loader(testing=False)

    # Load the pre-trained Siamese model
    siamese_model = load_model(SNN_PATH, custom_objects={'create_contrastive_loss': create_contrastive_loss})

    # Create and compile the Classification Model
    classifier_model = create_classification_model(siamese_model.get_layer(name="subnet"))

    # Train the Classification Model
    classifier_fit = classifier_model.fit(classify_train, epochs=epochs_classifier, validation_data=classify_val)

    # Save the trained Classification Model
    classifier_model.save(CLASSIFIER_PATH)

    # Create subplots for plotting
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Plot accuracy and validation accuracy with different colors
    axes[0].plot(snn_fit.history['accuracy'], label='accuracy', color='blue')
    axes[0].plot(snn_fit.history['val_accuracy'], label='val_accuracy', color='green')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].legend(loc='lower right')
    axes[0].set_title('Siamese Network Training Accuracy')

    # Plot loss and validation loss with different colors
    axes[1].plot(snn_fit.history['loss'], label='loss', color='red')
    axes[1].plot(snn_fit.history['val_loss'], label='val_loss', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_ylim([0, 1])
    axes[1].legend(loc='upper right')
    axes[1].set_title('Siamese Network Training Loss')

    # Plot validation accuracy
    axes[2].plot(classifier_fit.history['accuracy'], label='accuracy', color='blue')
    axes[2].plot(classifier_fit.history['val_accuracy'], label='val_accuracy', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_ylim([0, 1])
    axes[2].legend(loc='lower right')
    axes[2].set_title('Classification Model Training Accuracy')

    # Plot validation loss
    axes[3].plot(classifier_fit.history['loss'], label='loss', color='red')
    axes[3].plot(classifier_fit.history['val_loss'], label='val_loss', color='orange')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Loss')
    axes[3].set_ylim([0, 1])
    axes[3].legend(loc='upper right')
    axes[3].set_title('Classification Model Training Loss')

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Usage:
training(epochs_snn=40, epochs_classifier=20)